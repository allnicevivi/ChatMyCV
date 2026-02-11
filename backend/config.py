"""
Centralized Configuration

Loads and validates all environment variables for the ChatMyCV application.
Provides typed configuration objects for Azure OpenAI, Langfuse, Redis, and PostgreSQL.
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration"""
    api_key: str
    api_base: str
    api_version: str
    llm_engine: str
    llm_model: str
    embed_engine: str
    embed_model: str
    embed_dim: int
    embed_timeout: int

    @classmethod
    def from_env(cls) -> 'AzureOpenAIConfig':
        """Load configuration from environment variables"""
        return cls(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_base=os.getenv("AZURE_OPENAI_API_BASE", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            llm_engine=os.getenv("AZURE_OPENAI_LLM_ENGINE", ""),
            llm_model=os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4.1-mini"),
            embed_engine=os.getenv("AZURE_OPENAI_EMBED_ENGINE", ""),
            embed_model=os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-large"),
            embed_dim=int(os.getenv("EMBED_DIM", "1536")),
            embed_timeout=int(os.getenv("EMBED_TIMEOUT", "3"))
        )

    def validate(self) -> bool:
        """Validate that required fields are present"""
        required = [self.api_key, self.api_base, self.llm_engine, self.embed_engine]
        if not all(required):
            logger.error("Azure OpenAI configuration incomplete. Check .env file.")
            return False
        return True


@dataclass
class LangfuseConfig:
    """Langfuse observability configuration"""
    public_key: str
    secret_key: str
    host: str
    enabled: bool

    @classmethod
    def from_env(cls) -> 'LangfuseConfig':
        """Load configuration from environment variables"""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        enabled = bool(public_key and secret_key)

        return cls(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            enabled=enabled
        )


@dataclass
class RedisConfig:
    """Redis memory store configuration"""
    host: str
    port: int
    db: int
    password: Optional[str]
    enabled: bool

    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Load configuration from environment variables"""
        password = os.getenv("REDIS_PASSWORD", "")

        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=password if password else None,
            enabled=True  # Enabled by default, will fall back if connection fails
        )


@dataclass
class PostgresConfig:
    """PostgreSQL HITL database configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str
    enabled: bool

    @classmethod
    def from_env(cls) -> 'PostgresConfig':
        """Load configuration from environment variables"""
        user = os.getenv("POSTGRES_USER", "")
        password = os.getenv("POSTGRES_PASSWORD", "")
        database = os.getenv("POSTGRES_DB", "")

        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5433")),
            database=database,
            user=user,
            password=password,
            enabled=bool(user and password and database)
        )


@dataclass
class AppConfig:
    """Overall application configuration"""
    azure_openai: AzureOpenAIConfig
    langfuse: LangfuseConfig
    redis: RedisConfig
    postgres: PostgresConfig

    # App settings
    app_name: str = "ChatMyCV"
    app_version: str = "0.2.0"
    debug: bool = False

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load all configuration from environment"""
        return cls(
            azure_openai=AzureOpenAIConfig.from_env(),
            langfuse=LangfuseConfig.from_env(),
            redis=RedisConfig.from_env(),
            postgres=PostgresConfig.from_env(),
            debug=os.getenv("DEBUG", "False").lower() == "true"
        )

    def validate(self) -> bool:
        """Validate all configurations"""
        valid = True

        # Azure OpenAI is required
        if not self.azure_openai.validate():
            logger.error("Azure OpenAI configuration is invalid")
            valid = False

        # LLMOps components are optional but log warnings
        if not self.langfuse.enabled:
            logger.warning("Langfuse is not configured. Observability features will be disabled.")

        if not self.redis.enabled:
            logger.warning("Redis configuration incomplete. Will attempt connection with defaults.")

        if not self.postgres.enabled:
            logger.warning("PostgreSQL is not configured. HITL features will be disabled.")

        return valid

    def print_summary(self) -> None:
        """Print configuration summary (safe for logging)"""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        print(f"App:          {self.app_name} v{self.app_version}")
        print(f"Debug:        {self.debug}")
        print("\nAzure OpenAI:")
        print(f"  Endpoint:   {self.azure_openai.api_base}")
        print(f"  LLM Model:  {self.azure_openai.llm_model}")
        print(f"  Embed Model: {self.azure_openai.embed_model}")
        print("\nLangfuse:")
        print(f"  Enabled:    {self.langfuse.enabled}")
        print(f"  Host:       {self.langfuse.host if self.langfuse.enabled else 'N/A'}")
        print("\nRedis:")
        print(f"  Enabled:    {self.redis.enabled}")
        print(f"  Host:       {self.redis.host}:{self.redis.port}")
        print("\nPostgreSQL:")
        print(f"  Enabled:    {self.postgres.enabled}")
        print(f"  Database:   {self.postgres.database if self.postgres.enabled else 'N/A'}")
        print("="*60 + "\n")


# Global configuration instance
config = AppConfig.from_env()


# Export
__all__ = [
    'config',
    'AppConfig',
    'AzureOpenAIConfig',
    'LangfuseConfig',
    'RedisConfig',
    'PostgresConfig'
]


if __name__ == "__main__":
    # Test configuration loading
    cfg = AppConfig.from_env()
    cfg.print_summary()

    if cfg.validate():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
