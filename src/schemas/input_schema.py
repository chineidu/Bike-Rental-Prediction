from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field  # type: ignore
from pydantic.alias_generators import to_camel


def round_probability(value: float) -> float:
    """Round a float value to two decimal places.

    Returns:
        float: Rounded value.
    """
    if isinstance(value, float):
        return round(value, 2)
    return value


Float = Annotated[float, BeforeValidator(round_probability)]


class BaseSchema(BaseModel):
    """Base schema class that inherits from Pydantic BaseModel.

    This class provides common configuration for all schema classes including
    camelCase alias generation, population by field name, and attribute mapping.
    """

    model_config: ConfigDict = ConfigDict(  # type: ignore
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )


class _RawInputSchema(BaseSchema):
    """
    Schema for raw input data.
    """

    season: int = Field(
        ..., ge=1, le=4, description="Season (1: winter, 2: spring, 3: summer, 4: fall)"
    )
    mnth: int = Field(..., ge=1, le=12, description="Month (1 to 12)")
    holiday: int = Field(
        ..., ge=0, le=1, description="Whether the day is a holiday (0: no, 1: yes)"
    )
    hr: int = Field(..., ge=0, le=23, description="Hour of the day (0 to 23)")
    weekday: int = Field(
        ..., ge=0, le=6, description="Day of the week (0: Sunday to 6: Saturday)"
    )
    workingday: int = Field(
        ..., ge=0, le=1, description="Whether the day is a working day (0: no, 1: yes)"
    )
    weathersit: int = Field(..., ge=1, le=4, description="Weather situation (1 to 4)")
    temp: Float = Field(
        ..., ge=0.0, le=1.0, description="Normalized temperature in Celsius (0 to 1)"
    )
    hum: Float = Field(..., ge=0.0, le=1.0, description="Normalized humidity (0 to 1)")
    windspeed: Float = Field(
        ..., ge=0.0, le=1.0, description="Normalized wind speed (0 to 1)"
    )
    cnt: int = Field(..., ge=0, description="Count of bike rentals")
    datetime: str | None = Field(
        default=None, description="Datetime in the format 'YYYY-MM-DD HH:MM:SS'"
    )
    yr: int | None = Field(default=None, description="Year (0: 2011, 1: 2012, etc.)")


class RawInputSchema(BaseSchema):
    data: list[_RawInputSchema]

    class Config:
        from_attributes = True
        json_schema_extra: dict[str, list[Any]] = {
            "examples": [
                {
                    "data": [
                        {
                            "datetime": "2011-06-17 08:00:00",
                            "season": 2,
                            "yr": 0,
                            "mnth": 6,
                            "hr": 8,
                            "holiday": 0,
                            "weekday": 5,
                            "workingday": 1,
                            "weathersit": 2,
                            "temp": 0.6,
                            "hum": 0.83,
                            "windspeed": 0.0,
                            "cnt": 454,
                        }
                    ]
                }
            ]
        }
