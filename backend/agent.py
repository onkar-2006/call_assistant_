import sqlite3
from typing import Optional
from dataclasses import dataclass
from contextlib import contextmanager
import enum
import logging
from livekit.agents import JobContext, AgentSession, WorkerOptions, cli, Agent, function_tool, RunContext
from livekit.plugins import groq, silero
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# Database setup
@dataclass
class Car:
    vin: str
    make: str
    model: str
    year: int

class DatabaseDriver:
    def __init__(self, db_path: str = "auto_db.sqlite"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cars (
                    vin TEXT PRIMARY KEY,
                    make TEXT NOT NULL,
                    model TEXT NOT NULL,
                    year INTEGER NOT NULL
                )
            """)
            conn.commit()

    def create_car(self, vin: str, make: str, model: str, year: int) -> Car:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO cars (vin, make, model, year) VALUES (?, ?, ?, ?)",
                (vin, make, model, year)
            )
            conn.commit()
            return Car(vin=vin, make=make, model=model, year=year)

    def get_car_by_vin(self, vin: str) -> Optional[Car]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cars WHERE vin = ?", (vin,))
            row = cursor.fetchone()
            if not row:
                return None
            return Car(vin=row[0], make=row[1], model=row[2], year=row[3])

# Assistant logic
logger = logging.getLogger("user-data")
logger.setLevel(logging.INFO)

DB = DatabaseDriver()

class CarDetails(enum.Enum):
    VIN = "vin"
    Make = "make"
    Model = "model"
    Year = "year"

class AssistantFnc:
    def __init__(self):
        self._car_details = {
            CarDetails.VIN: "",
            CarDetails.Make: "",
            CarDetails.Model: "",
            CarDetails.Year: ""
        }

    def get_car_str(self):
        return "\n".join([f"{key.value}: {value}" for key, value in self._car_details.items()])

    def lookup_car(self, vin: str) -> str:
        logger.info("lookup car - vin: %s", vin)
        result = DB.get_car_by_vin(vin)
        if result is None:
            return "Car not found"
        self._car_details = {
            CarDetails.VIN: result.vin,
            CarDetails.Make: result.make,
            CarDetails.Model: result.model,
            CarDetails.Year: result.year
        }
        return f"The car details are:\n{self.get_car_str()}"

    def create_car(self, vin: str, make: str, model: str, year: int) -> str:
        logger.info("create car - vin: %s, make: %s, model: %s, year: %s", vin, make, model, year)
        result = DB.create_car(vin, make, model, year)
        if result is None:
            return "Failed to create car"
        self._car_details = {
            CarDetails.VIN: result.vin,
            CarDetails.Make: result.make,
            CarDetails.Model: result.model,
            CarDetails.Year: result.year
        }
        return "Car profile created!"

    def get_car_details(self) -> str:
        logger.info("get car details")
        if not self._car_details[CarDetails.VIN]:
            return "No car profile set yet."
        return f"Current car:\n{self.get_car_str()}"

    def has_car(self) -> bool:
        return bool(self._car_details[CarDetails.VIN])

assistant_fnc = AssistantFnc()

INSTRUCTIONS = """
You are the manager of a call center speaking to a customer. Your goal is to help answer their questions or direct them to the correct department.
Start by collecting or looking up their car information. Once you have the car information, you can answer their questions or direct them appropriately.
"""

WELCOME_MESSAGE = """
Welcome to the Auto Service Center! Please provide the VIN of your vehicle to lookup your profile. If you don't have a profile, please say 'create profile'.
"""

LOOKUP_VIN_MESSAGE = lambda msg: f"""If the user has provided a VIN, attempt to look it up.
If the VIN does not exist, create the entry in the database using your tools.
If the user doesn't have a VIN, ask them for details required to create a new car.
User message: {msg}
"""

class VehicleRegistrationAgent(Agent):
    def __init__(self):
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions=WELCOME_MESSAGE)

    @function_tool()
    async def lookup_car(self, context: RunContext, vin: str):
        """Look up a car by its VIN."""
        return assistant_fnc.lookup_car(vin)

    @function_tool()
    async def create_car(self, context: RunContext, vin: str, make: str, model: str, year: int):
        """Create a new car profile."""
        return assistant_fnc.create_car(vin, make, model, year)

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = VehicleRegistrationAgent()

    session = AgentSession(
        stt=groq.STT(),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=groq.TTS(model="playai-tts"),
        vad=silero.VAD.load(),
    )

    await session.start(room=ctx.room, agent=agent)

    @session.on("user_message")
    async def handle_msg(msg: str):
        msg_lower = msg.lower()
        if "lookup" in msg_lower and "vin" in msg_lower:
            # Expecting "lookup vin <VIN>"
            parts = msg_lower.split()
            try:
                vin_index = parts.index("vin") + 1
                vin = parts[vin_index]
                response = assistant_fnc.lookup_car(vin)
            except (ValueError, IndexError):
                response = "Please provide a VIN after the word 'vin' to look up."
        elif "create profile" in msg_lower:
            # Expecting "create profile VIN MAKE MODEL YEAR"
            parts = msg.split()
            try:
                vin = parts[2]
                make = parts[3]
                model = parts[4]
                year = int(parts[5])
                response = assistant_fnc.create_car(vin, make, model, year)
            except (IndexError, ValueError):
                response = "To create a profile, please say: create profile VIN MAKE MODEL YEAR"
        elif assistant_fnc.has_car():
            response = assistant_fnc.get_car_details()
        else:
            response = "Please provide your car's VIN to look up your profile, or say 'create profile' to register a new vehicle."

        await session.send_message(response)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
