from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables (.env)
load_dotenv(override=True)

# 2. Define the Pydantic Schema
class MovieSchema(BaseModel):
    title: str = Field(description="Title of the movie")
    release_year: Optional[int] = Field(
        default=None,
        description="Release year of the movie"
    )
    genre: List[str] = Field(
        description="List of movie genres"
    )
    director: Optional[str] = Field(
        default=None,
        description="Name of the director"
    )
    cast: List[str] = Field(
        description="List of main cast members"
    )
    rating: Optional[float] = Field(
        default=None,
        description="IMDb rating or score"
    )
    summary: str = Field(
        description="A short summary of the plot"
    )

# 3. Initialize Chat Model and Bind Schema
llm = init_chat_model(
    "gemini-1.5-flash",
    model_provider="google_genai",
    temperature=0.0
)

structured_llm = llm.with_structured_output(MovieSchema)

# 4. Create Prompt Template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert movie data extraction assistant. "
        "Extract structured movie information from the provided text. "
        "If information is missing, set it to null. "
        "For missing lists, return empty lists."
    ),
    (
        "human",
        "Extract information from this movie paragraph:\n\n{paragraph}"
    )
])

# 5. Build LCEL Chain
chain = prompt | structured_llm

# 6. Sample Input Paragraph
sample_paragraph = """
Interstellar is a 2014 epic science fiction film directed by
Christopher Nolan. It stars Matthew McConaughey, Anne Hathaway,
Jessica Chastain, and Bill Irwin.

The film follows a group of astronauts who travel through a wormhole
near Saturn in search of a new home for mankind.
It holds an IMDb rating of 8.7.
"""

# 7. Invoke Chain
result: MovieSchema = chain.invoke({
    "paragraph": sample_paragraph
})

# 8. Print Output
print("=== PARSED PYTHON OBJECT ===")
print("Title:", result.title)
print("Year:", result.release_year)
print("Genres:", result.genre)
print("Director:", result.director)
print("Cast:", result.cast)
print("Rating:", result.rating)
print("Summary:", result.summary)

print("\n=== CONVERTED TO CLEAN JSON ===")
print(result.model_dump_json(indent=2))