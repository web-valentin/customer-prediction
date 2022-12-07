import constants as constant


class WeatherDay:
    temperature = {
        "morning": 0.0,
        "midday": 0.0,
        "afternoon": 0.0,
        "average": 0.0,
        "total": 0.0
    }

    weather_main = {
        "morning": 0,
        "midday": 0,
        "afternoon": 0,
    }

    clouds_all = {
        "morning": 0.0,
        "midday": 0.0,
        "afternoon": 0.0,
        "average": 0.0,
        "total": 0.0
    }

    wind_speed = {
        "morning": 0.0,
        "midday": 0.0,
        "afternoon": 0.0,
        "average": 0.0,
        "total": 0.0
    }

    humidity = {
        "morning": 0.0,
        "midday": 0.0,
        "afternoon": 0.0,
        "average": 0.0,
        "total": 0.0
    }

    rain = {
        "morning": 0.0,
        "midday": 0.0,
        "afternoon": 0.0,
        "average": 0.0,
        "total": 0.0
    }

    snow = {
        "morning": 0.0,
        "midday": 0.0,
        "afternoon": 0.0,
        "average": 0.0,
        "total": 0.0
    }

    date = ""

    def __init__(self):
        self.name = "s"

    def get_weather_as_number(self, weather_as_string, daytime):
        weather = constant.MAIN_WEATHER
        if weather_as_string in weather:
            self.weather_main[daytime] = weather[weather_as_string]
            # self.weather_main[daytime] = weather_as_string
            return self.weather_main[daytime]
        else:
            print(weather_as_string)

    def reset_variables(self):
        self.temperature = {
            "morning": 0.0,
            "midday": 0.0,
            "afternoon": 0.0,
            "average": 0.0,
            "total": 0.0
        }

        self.weather_main = {
            "morning": 0,
            "midday": 0,
            "afternoon": 0,
        }

        self.clouds_all = {
            "morning": 0.0,
            "midday": 0.0,
            "afternoon": 0.0,
            "average": 0.0,
            "total": 0.0
        }

        self.wind_speed = {
            "morning": 0.0,
            "midday": 0.0,
            "afternoon": 0.0,
            "average": 0.0,
            "total": 0.0
        }

        self.humidity = {
            "morning": 0.0,
            "midday": 0.0,
            "afternoon": 0.0,
            "average": 0.0,
            "total": 0.0
        }

        self.rain = {
            "morning": 0.0,
            "midday": 0.0,
            "afternoon": 0.0,
            "average": 0.0,
            "total": 0.0
        }

        self.snow = {
            "morning": 0.0,
            "midday": 0.0,
            "afternoon": 0.0,
            "average": 0.0,
            "total": 0.0
        }
