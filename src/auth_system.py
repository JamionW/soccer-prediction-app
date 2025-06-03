import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import bcrypt
import jwt 
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from databases import Database
import httpx
import logging

logger = logging.getLogger(__name__)

# Configuration from environment variables
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# Security scheme for FastAPI
security = HTTPBearer()

class AuthManager:
    """
    Manages all authentication operations for the application.
    
    This class handles:
    - User registration with username/password
    - Password hashing using bcrypt (industry standard)
    - JWT token generation and validation
    - OAuth flow for Google and GitHub
    """
    
    def __init__(self, database: Database):
        self.db = database
        
    async def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """
        Register a new user with username and password.
        
        Args:
            username: Unique username
            email: User's email address
            password: Plain text password (will be hashed)
            
        Returns:
            Dictionary containing user info and JWT token
        """
        # Check if username or email already exists
        existing_user = await self.db.fetch_one(
            "SELECT user_id FROM users WHERE username = :username OR email = :email",
            values={"username": username, "email": email}
        )
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        # Hash the password using bcrypt
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create the user
        query = """
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (:username, :email, :password_hash, NOW())
            RETURNING user_id, username, email, is_active, is_admin
        """
        
        user = await self.db.fetch_one(query, values={
            "username": username,
            "email": email,
            "password_hash": password_hash.decode('utf-8')
        })
        
        # Generate JWT token for immediate login
        token = self.generate_jwt_token(user['user_id'])
        
        # Store token hash in database
        await self._store_token(user['user_id'], token)
        
        return {
            "user": dict(user),
            "token": token,
            "token_type": "Bearer"
        }
    
    async def login_user(self, username_or_email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate a user with username/email and password.
        
        Args:
            username_or_email: Username or email
            password: Plain text password
            
        Returns:
            Dictionary containing user info and JWT token
        """
        # Find user by username or email
        query = """
            SELECT user_id, username, email, password_hash, is_active, is_admin
            FROM users
            WHERE (username = :identifier OR email = :identifier)
            AND password_hash IS NOT NULL
        """
        
        user = await self.db.fetch_one(
            query, 
            values={"identifier": username_or_email}
        )
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Check if user is active
        if not user['is_active']:
            raise HTTPException(status_code=403, detail="Account is disabled")
        
        # Verify password using bcrypt
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login time
        await self.db.execute(
            "UPDATE users SET last_login_at = NOW() WHERE user_id = :user_id",
            values={"user_id": user['user_id']}
        )
        
        # Generate JWT token
        token = self.generate_jwt_token(user['user_id'])
        
        # Store token hash
        await self._store_token(user['user_id'], token)
        
        return {
            "user": {
                "user_id": user['user_id'],
                "username": user['username'],
                "email": user['email'],
                "is_admin": user['is_admin']
            },
            "token": token,
            "token_type": "Bearer"
        }
    
    def generate_jwt_token(self, user_id: int) -> str:
        """
        Generate a JWT token for a user.
        
        JWT tokens contain:
        - user_id: The user's ID
        - exp: Expiration timestamp
        - iat: Issued at timestamp
        """
        payload = {
            "user_id": user_id,
            "exp": datetime.now(datetime.timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
            "iat": datetime.now(datetime.timezone.utc)
        }
        
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    async def validate_token(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
        """
        Validate a JWT token from the Authorization header.
        
        This is used as a dependency in FastAPI routes to ensure authentication.
        """
        token = credentials.credentials
        
        try:
            # Decode the JWT token
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("user_id")
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Check if token exists and is not revoked
            token_hash = self._hash_token(token)
            stored_token = await self.db.fetch_one(
                """
                SELECT * FROM auth_tokens 
                WHERE token_hash = :token_hash 
                AND is_revoked = false 
                AND expires_at > NOW()
                """,
                values={"token_hash": token_hash}
            )
            
            if not stored_token:
                raise HTTPException(status_code=401, detail="Token not found or expired")
            
            # Get user information
            user = await self.db.fetch_one(
                "SELECT user_id, username, email, is_active, is_admin FROM users WHERE user_id = :user_id",
                values={"user_id": user_id}
            )
            
            if not user or not user['is_active']:
                raise HTTPException(status_code=401, detail="User not found or inactive")
            
            return dict(user)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def _store_token(self, user_id: int, token: str):
        """Store a token hash in the database."""
        token_hash = self._hash_token(token)
        
        # JWT already contains expiration, extract it
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        expires_at = datetime.fromtimestamp(payload['exp'])
        
        await self.db.execute(
            """
            INSERT INTO auth_tokens (user_id, token_hash, expires_at, created_at)
            VALUES (:user_id, :token_hash, :expires_at, NOW())
            """,
            values={
                "user_id": user_id,
                "token_hash": token_hash,
                "expires_at": expires_at
            }
        )
    
    def _hash_token(self, token: str) -> str:
        """Hash a token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    # OAuth Methods
    
    async def oauth_login(self, provider: str, code: str) -> Dict[str, Any]:
        """
        Handle OAuth login for Google or GitHub.
        
        Args:
            provider: 'google' or 'github'
            code: Authorization code from OAuth provider
            
        Returns:
            Dictionary containing user info and JWT token
        """
        if provider == 'google':
            user_info = await self._get_google_user_info(code)
        elif provider == 'github':
            user_info = await self._get_github_user_info(code)
        else:
            raise HTTPException(status_code=400, detail="Invalid OAuth provider")
        
        # Check if user exists with this OAuth provider
        oauth_user = await self.db.fetch_one(
            """
            SELECT u.* FROM users u
            JOIN oauth_providers op ON u.user_id = op.user_id
            WHERE op.provider_name = :provider AND op.provider_user_id = :provider_user_id
            """,
            values={
                "provider": provider,
                "provider_user_id": user_info['id']
            }
        )
        
        if oauth_user:
            # Existing user, update last login
            user_id = oauth_user['user_id']
            await self.db.execute(
                "UPDATE users SET last_login_at = NOW() WHERE user_id = :user_id",
                values={"user_id": user_id}
            )
        else:
            # New user, create account
            # Generate a unique username from email
            base_username = user_info['email'].split('@')[0]
            username = await self._generate_unique_username(base_username)
            
            # Create user without password
            user_result = await self.db.fetch_one(
                """
                INSERT INTO users (username, email, created_at, last_login_at)
                VALUES (:username, :email, NOW(), NOW())
                RETURNING user_id
                """,
                values={
                    "username": username,
                    "email": user_info['email']
                }
            )
            
            user_id = user_result['user_id']
            
            # Create OAuth provider record
            await self.db.execute(
                """
                INSERT INTO oauth_providers (user_id, provider_name, provider_user_id, created_at)
                VALUES (:user_id, :provider, :provider_user_id, NOW())
                """,
                values={
                    "user_id": user_id,
                    "provider": provider,
                    "provider_user_id": user_info['id']
                }
            )
        
        # Generate JWT token
        token = self.generate_jwt_token(user_id)
        await self._store_token(user_id, token)
        
        # Get full user info
        user = await self.db.fetch_one(
            "SELECT user_id, username, email, is_admin FROM users WHERE user_id = :user_id",
            values={"user_id": user_id}
        )
        
        return {
            "user": dict(user),
            "token": token,
            "token_type": "Bearer"
        }
    
    async def _get_google_user_info(self, code: str) -> Dict[str, Any]:
        """Exchange Google authorization code for user info."""
        async with httpx.AsyncClient() as client:
            # Exchange code for access token
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
                    "grant_type": "authorization_code"
                }
            )
            
            if token_response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to get Google access token")
            
            access_token = token_response.json()["access_token"]
            
            # Get user info
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if user_response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to get Google user info")
            
            return user_response.json()
    
    async def _get_github_user_info(self, code: str) -> Dict[str, Any]:
        """Exchange GitHub authorization code for user info."""
        async with httpx.AsyncClient() as client:
            # Exchange code for access token
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": code
                },
                headers={"Accept": "application/json"}
            )
            
            if token_response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to get GitHub access token")
            
            access_token = token_response.json()["access_token"]
            
            # Get user info
            user_response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if user_response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to get GitHub user info")
            
            user_data = user_response.json()
            
            # Get primary email if not public
            if not user_data.get('email'):
                email_response = await client.get(
                    "https://api.github.com/user/emails",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if email_response.status_code == 200:
                    emails = email_response.json()
                    primary_email = next((e['email'] for e in emails if e['primary']), None)
                    if primary_email:
                        user_data['email'] = primary_email
            
            return user_data
    
    async def _generate_unique_username(self, base_username: str) -> str:
        """Generate a unique username by appending numbers if needed."""
        username = base_username
        counter = 1
        
        while True:
            exists = await self.db.fetch_one(
                "SELECT 1 FROM users WHERE username = :username",
                values={"username": username}
            )
            
            if not exists:
                return username
            
            username = f"{base_username}{counter}"
            counter += 1
    
    async def logout(self, token: str):
        """Revoke a JWT token."""
        token_hash = self._hash_token(token)
        
        await self.db.execute(
            "UPDATE auth_tokens SET is_revoked = true WHERE token_hash = :token_hash",
            values={"token_hash": token_hash}
        )
    
    async def cleanup_expired_tokens(self):
        """Remove expired tokens from the database (maintenance task)."""
        await self.db.execute(
            "DELETE FROM auth_tokens WHERE expires_at < NOW() OR is_revoked = true"
        )