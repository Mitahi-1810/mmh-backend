from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.config import get_settings

# auto_error=False → returns None when the Authorization header is absent
bearer         = HTTPBearer(auto_error=True)   # strict — for protected endpoints
optional_bearer = HTTPBearer(auto_error=False)  # lenient — for guest-accessible endpoints


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            get_settings().supabase_jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def get_user_id(payload: dict = Depends(verify_token)) -> str:
    """Strict auth — endpoint requires a valid JWT."""
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing user ID in token")
    return user_id


def get_optional_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer),
) -> str | None:
    """
    Soft auth — returns the user-id string if a valid JWT is present,
    or None if no Authorization header was sent (guest / anonymous request).
    An *invalid* token (present but malformed/expired) is silently treated as
    anonymous so the request still proceeds rather than failing with 401.
    """
    if credentials is None:
        return None
    try:
        payload = jwt.decode(
            credentials.credentials,
            get_settings().supabase_jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        return payload.get("sub")
    except JWTError:
        return None
