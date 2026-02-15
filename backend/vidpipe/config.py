"""Configuration management with YAML and environment variable support."""

from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that loads configuration from YAML file."""

    def get_field_value(self, field, field_name: str):
        # Not used with prepare method
        pass

    def prepare_field_value(self, field_name: str, field, value, value_is_complex: bool):
        return value

    def __call__(self):
        # Load from config.yaml in current directory
        yaml_path = Path("config.yaml")
        if not yaml_path.exists():
            return {}

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        return data


class GoogleCloudConfig(BaseModel):
    """Google Cloud configuration.

    project_id is required and must be set via .env or environment variable.
    """

    project_id: str
    location: str = "us-central1"
    use_vertex_ai: bool = True


class ModelsConfig(BaseModel):
    """AI model identifiers."""

    storyboard_llm: str
    image_gen: str
    image_conditioned: str
    video_gen: str


class PipelineConfig(BaseModel):
    """Pipeline execution parameters."""

    default_style: str
    default_aspect_ratio: str
    default_clip_duration: int
    max_scenes: int
    image_gen_delay: int
    video_poll_interval: int
    video_poll_max: int
    video_gen_concurrency: int
    crossfade_seconds: float
    retry_max_attempts: int
    retry_base_delay: int


class StorageConfig(BaseModel):
    """Storage and database configuration."""

    database_url: str
    tmp_dir: Path

    @field_validator("tmp_dir", mode="before")
    @classmethod
    def convert_tmp_dir_to_path(cls, v):
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str
    port: int


class Settings(BaseSettings):
    """Main application settings with YAML and environment variable support.

    Configuration sources (in priority order):
    1. Environment variables (prefix: VIDPIPE_, delimiter: __)
    2. YAML file (config.yaml)
    3. Field defaults
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="VIDPIPE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_cloud: GoogleCloudConfig
    models: ModelsConfig
    pipeline: PipelineConfig
    storage: StorageConfig
    server: ServerConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        """Customize settings sources to include YAML configuration.

        Priority order (highest to lowest):
        1. Environment variables
        2. YAML file
        3. Init settings (programmatic defaults)
        """
        return (
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            init_settings,
        )


# Singleton instance
settings = Settings()
