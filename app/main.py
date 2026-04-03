from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from app.database import engine, get_db
from app import models, schemas
from app.auth import hash_password, verify_password, create_access_token
from app.routers import papers, rag, compare

# Create all DB tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Research Assistant API",
    description="Bohrium-level RAG research platform",
    version="1.0.0"
)

# CORS — allow Lovable frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down to Lovable URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(papers.router)
app.include_router(rag.router)
app.include_router(compare.router)


# ── Auth endpoints ──

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.post("/auth/register", response_model=schemas.UserOut)
def register(body: schemas.RegisterRequest, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = models.User(
        name     = body.name,
        email    = body.email,
        password = hash_password(body.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=schemas.TokenResponse)
def login(body: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == body.email).first()
    if not user or not verify_password(body.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token}

@app.get("/")
def root():
    return {"status": "AI Research Assistant API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}