"""
Authentication router — login, refresh, user profile.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from api.repositories.user_repo import UserRepository

from api.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user,
)
from api.database import get_db, User, AuditLog
from api.models import LoginRequest, TokenResponse, RefreshRequest, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT tokens."""
    repo = UserRepository(db)
    user = await repo.get_by_email(request.email)

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Compte désactivé. Contactez l'administrateur.",
        )

    await repo.update_last_login(user)
    await db.commit()

    # Create tokens
    token_data = {
        "sub": user.id,
        "email": user.email,
        "department": user.department_id,
        "role": user.role,
    }
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    # Store refresh token in DB
    from datetime import timedelta
    from api.config import get_settings
    settings = get_settings()
    expire_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    await repo.save_refresh_token(refresh_token, user.id, expire_at)

    # Audit log
    db.add(AuditLog(
        user_id=user.id,
        action="login",
        department_id=user.department_id,
        metadata_={"ip": "local"},
    ))
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.model_validate(user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Get a new access token using a refresh token."""
    payload = decode_token(request.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de rafraîchissement invalide",
        )

    user_id = payload.get("sub")
    
    repo = UserRepository(db)
    
    # Check if refresh token is in DB
    db_token = await repo.get_refresh_token(request.refresh_token)
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de rafraîchissement révoqué ou introuvable",
        )
    
    # Delete the used refresh token (rotate it)
    await repo.delete_refresh_token(db_token)

    user = await repo.get_by_id(user_id)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur introuvable ou désactivé",
        )

    token_data = {
        "sub": user.id,
        "email": user.email,
        "department": user.department_id,
        "role": user.role,
    }
    new_access = create_access_token(token_data)
    new_refresh = create_refresh_token(token_data)

    # Store the new refresh token
    from datetime import timedelta
    from api.config import get_settings
    settings = get_settings()
    expire_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    await repo.save_refresh_token(new_refresh, user.id, expire_at)
    
    await db.commit()

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        user=UserResponse.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return UserResponse.model_validate(current_user)


@router.post("/logout", status_code=204)
async def logout(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db)
):
    """Revoke a refresh token (logout)."""
    repo = UserRepository(db)
    db_token = await repo.get_refresh_token(request.refresh_token)
    
    if db_token:
        await repo.delete_refresh_token(db_token)
        await db.commit()
