from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PageScrape(BaseModel):
    id: str
    source: str | None = None
    type: str |None = None
    content: str |None =None
    gptkey: str |None =None
    namespace: str |None =None
    webhooks: str |None =None

@app.post("/api/scrape/single")
async def create_screape_item(item: PageScrape):
    # Add logic to process and store the item data
    return {"message": f"Item {item.id} created successfully"}

def main(): 
    import uvicorn
    uvicorn.run("tilellm.__main__:app", host="0.0.0.0", port=8000, reload=True) 

if __name__ == "__main__":
   main()
