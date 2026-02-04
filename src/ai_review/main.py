from fastapi import FastAPI

app = FastAPI(title="AI Review Manager")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
