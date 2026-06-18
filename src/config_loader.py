# -------------------------------------------------------------------
# config_loader.py
# Loads and validates application configuration from YAML file
# and environment variables.
# -------------------------------------------------------------------

import os
import logging
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Module-level logger for configuration loading events
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads application configuration from YAML file with environment variable overrides.

    This class reads the settings.yaml file and allows environment variables
    prefixed with LLM_CACHE_ to override any configuration value.

    Attributes:
        config_path: Path to the YAML configuration file.
        config: The loaded configuration dictionary.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the configuration loader.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = config_path  # Store the path to the config file
        self.config: Dict[str, Any] = {}  # Initialize empty config dictionary

        # Load environment variables from .env file if present
        load_dotenv()

        # Load the configuration from file
        self._load_config()

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Validate required configuration values
        self._validate_config()

    def _load_config(self) -> None:
        """Load configuration from the YAML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        try:
            # Open and parse the YAML configuration file
            with open(self.config_path, "r", encoding="utf-8") as config_file:
                self.config = yaml.safe_load(config_file)

            logger.info("Configuration loaded from %s", self.config_path)

        except FileNotFoundError:
            # Log the error and re-raise for the caller to handle
            logger.error("Configuration file not found: %s", self.config_path)
            raise

        except yaml.YAMLError as yaml_error:
            # Log the parsing error and re-raise
            logger.error("Error parsing YAML config: %s", yaml_error)
            raise

    def _apply_env_overrides(self) -> None:
        """Override configuration values with environment variables.

        Environment variables prefixed with LLM_CACHE_ will override config values.
        Nested keys use double underscore as separator.
        Example: LLM_CACHE_LLM__API_KEY overrides config['llm']['api_key']
        """
        # Iterate through all environment variables
        for env_key, env_value in os.environ.items():
            # Only process variables with our prefix
            if env_key.startswith("LLM_CACHE_"):
                # Remove the prefix and convert to lowercase
                config_key = env_key[len("LLM_CACHE_"):].lower()

                # Split on double underscore for nested keys
                key_parts = config_key.split("__")

                # Navigate to the correct nested dictionary level
                current_level = self.config
                for part in key_parts[:-1]:
                    if part in current_level:
                        current_level = current_level[part]
                    else:
                        # Create missing intermediate dictionaries
                        current_level[part] = {}
                        current_level = current_level[part]

                # Set the final value at the deepest level
                current_level[key_parts[-1]] = env_value
                logger.debug("Config override applied from env: %s", env_key)

    def _validate_config(self) -> None:
        """Validate that all required configuration sections are present.

        Raises:
            ValueError: If a required configuration section is missing.
        """
        # Define required top-level configuration sections
        required_sections = ["cache", "similarity", "llm", "logging"]

        for section in required_sections:
            if section not in self.config:
                error_message = f"Missing required config section: {section}"
                logger.error(error_message)
                raise ValueError(error_message)

        logger.info("Configuration validation passed")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by section and key.

        Args:
            section: The top-level configuration section name.
            key: The key within the section.
            default: Default value if key is not found.

        Returns:
            The configuration value, or the default if not found.
        """
        # Navigate to the section and retrieve the key
        section_data = self.config.get(section, {})
        return section_data.get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Retrieve an entire configuration section.

        Args:
            section: The top-level configuration section name.

        Returns:
            Dictionary containing all key-value pairs in the section.
        """
        return self.config.get(section, {})
