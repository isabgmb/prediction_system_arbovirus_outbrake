"""
api/controllers/auth.py
------------------------
POST /auth/token  — OAuth2 password flow (RN07).
JWT middleware intercepts all authenticated requests before they reach controllers.
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

from src.core.config import settings
from src.api.models.schemas import TokenOut

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# In production, replace with a real user lookup against the DB
FAKE_USERS = {
    "admin": "$2b$12$placeholder_hash"  # bcrypt hash of actual password
}


def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """FastAPI dependency — validates JWT and returns username."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key,
                             algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username


@router.post("/token", response_model=TokenOut)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenOut:
    """
    Authenticate and return a JWT access token.
    The frontend calls this via POST /auth/token with username + password.
    """
    if form_data.username not in FAKE_USERS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )
    # TODO: replace with real bcrypt verification against DB
    token = create_access_token({"sub": form_data.username})
    return TokenOut(access_token=token)
