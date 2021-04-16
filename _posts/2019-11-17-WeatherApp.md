```python
"""
    File: Final Week 12.py
    Author: Aritzi Piedras-Silva
    Date: 11/17/2019
    Course: DSC 510 - Introduction to programing
    Desc: This program will run through a webservice in order to get current weather
    using zip code or a city name followed with the state. The output will be listed in 3 columns.
    The first column will contain the dates.
    The second column will contains the temperatures in degrees.
    The third will contains the humidity in percentage.
"""


import requests


class weatherMapConnector:

    def __init__(self, userInput: list, userInputIdentifier: str):
        #   THe URL, API Key, Units and Country will remain the same for every connection
        #   to the webservice
        self.URL = 'http://api.openweathermap.org/data/2.5/'
        self.apiKey = '5271ded7a96b73ac693686bcbdfde5b4'
        self.units = 'imperial'
        self.country = 'us'

        #   if the user chose to look up the weather using a zip code
        if userInputIdentifier == 'zip':
            self.zip = userInput[0]
            try:
                self.displayWeatherAndForecast(self.weatherByZip(), self.forecastByZip())
            except:
                print('Either we could not connect to the web service, or you entered an invalid zip code!\n')
                print('If this problem persists, please try again later!')
                return
        #   if the user chose to look up the weather using the city and state
        elif userInputIdentifier == 'city':
            self.city = userInput[0]
            self.state = userInput[1]
            try:
                self.displayWeatherAndForecast(self.weatherByCity(), self.forecastByCity())
            except:
                print('Either we could not connect to the web service, or you entered an invalid city and state combination!\n')
                print('If this problem persists, please try again later!')
                return

    #   a function to call the web service to find the current weather statistics by zip code

    def weatherByZip(self):
        try:
            weatherResponse = requests.get('{baseURL}weather?zip={zipcode}&units={units}&appid={apikey}'.format(baseURL=self.URL, zipcode=self.zip, units=self.units, apikey=self.apiKey))
            #   check that we did not get a 404 error
            assert int(weatherResponse.json()['cod']) < 400
            return weatherResponse.json()
        except AssertionError:
            return
        except requests.exceptions.RequestException as e:
            return

    #   a function to call the web service to find the current weather statistics by city and state
    def forecastByZip(self):
        try:
            weatherResponse = requests.get('{baseURL}forecast?zip={zipcode}&units={units}&appid={apikey}'.format(baseURL=self.URL, zipcode=self.zip, units=self.units, apikey=self.apiKey))
            assert int(weatherResponse.json()['cod']) < 400
            return weatherResponse.json()
        except AssertionError:
            return
        except requests.exceptions.RequestException as e:
            return

    #   a function to call the web service to find the future weather statistics by zip code
    def weatherByCity(self):
        try:
            weatherResponse = requests.get('{baseURL}weather?q={city},{state},{country}&units={units}&appid={apikey}'.format(baseURL=self.URL, city=self.city,state=self.state, country=self.country, units=self.units, apikey=self.apiKey))
            assert int(weatherResponse.json()['cod']) < 400
            return weatherResponse.json()
        except AssertionError:
            return
        except requests.exceptions.RequestException as e:
            return

    #   a function to call the web service to find the future weather statistics
    def forecastByCity(self):
        try:
            weatherResponse = requests.get('{baseURL}forecast?q={city},{state},{country}&units={units}&appid={apikey}'.format(baseURL=self.URL, city=self.city,state=self.state, units=self.units, country=self.country, apikey=self.apiKey))
            assert int(weatherResponse.json()['cod']) < 400
            return weatherResponse.json()
        except AssertionError:
            return
        except requests.exceptions.RequestException as e:
            return

    def displayWeatherAndForecast(self, weather: dict, forecast: dict):
        qualWeather = weather['weather']
        quantWeather = weather['main']
        locationName = weather['name']
        windSpeed = weather['wind']['speed']
        windDirection = self.convertDegrees(weather['wind']['deg'])

        print('The current weather in ', locationName, ':\n\n')
        print(qualWeather[0]['main'], ', ', qualWeather[0]['description'], '\n')
        print('The temperature is currently ', quantWeather['temp'], ' and ', quantWeather['humidity'], '% humidity\n')
        print('The wind is blowing, ', windDirection, ' at ', windSpeed, ' MPH\n')

        print('Over the next couple days you can expect the forecast to look like: \n\n')

        forecastDict = {
            "qual": {},
            "temperature": {},
            "humidity": {}
        }
        forecastList = forecast['list']
        print('     DATE                         WEATHER            DEGREES       HUMIDITY  ')
        for i in forecastList:
            # print(i)

            qual = i['weather'][0]['description']
            temperature = i['main']['temp']
            humidity = i['main']['humidity']
            date = i['dt_txt']
            print(date, '        ', qual, '        ', temperature, '        ', humidity, '%\n')

    #   Function used to convert the circular degrees received from the weather map web service
    #   to the direction that corresponds to that degree amount
    def convertDegrees(self, degrees: int) -> str:
        if degrees >= 0 and degrees <= 22.5 or degrees >= 337.5 and degrees < 360:
            return 'N'
        elif degrees > 22.5 and degrees < 67.5:
            return 'NE'
        elif degrees >= 67.5 and degrees <= 112.5:
            return 'E'
        elif degrees > 112.5 and degrees < 157.5:
            return 'SE'
        elif degrees >= 157.5 and degrees <= 202.5:
            return 'S'
        elif degrees > 202.5 and degrees < 247.5:
            return 'SW'
        elif degrees >= 247.5 and degrees <= 292.5:
            return 'W'
        else:
            return 'NW'


def main():
    print('Welcome to the Aritzi Silva Weather Service, bringing you the most up to date weather patterns and futurecasts occurring in your area!\n')

    while True:
        print('If you would like to look up weather data by zip code, please enter the word \'zip\'\n')
        print('If you would like to look up weather data by city and state please enter the word \'city\''
              ' to quit enter \'q\'. \n')
        #   Variable which stores the method the user would like to use to look up their weather data
        infoType = input()
        #   List to hold either just the zip code or the city and state combination
        userInputList = []
        start = False
        if infoType == 'zip':
            zipcode = input('Please enter a valid zip code in the United States \n')
            #   check that the zipcode entered is made up of 5 characters and they are all numbers
            if len(zipcode) == 5 and zipcode.isnumeric():
                userInputList.append(zipcode)
                start = True
            else:
                print('Uh oh! Looks like you entered a non-valid zipcode. Remember, the zip code should be a valid zip code in the US and should consist of 5 digits!')
        elif infoType == 'city':
            city = input('Please enter a valid city in the United States \n')
            state = input('What state is that city in? Please use the 2 letter format: NY, CA, FL, etc. \n')
            #   check that the city and state are both made up of letters only and the length of the
            #   state code is exactly 2 characters
            if city.isalpha() and state.isalpha() and len(state) == 2:
                userInputList.append(city)
                userInputList.append(state)
                start = True
            else:
                print('Uh oh! Looks like you entered something in the wrong format! Remember, city should be a valid city name in the US made up of all letters and state should be the two letter state code, like NY for New York or FL for Florida')
        elif infoType == 'q':
            print('Thank you for using my weather program and for a great semester!')
            return
        else:
            print('Uh oh! Looks like you entered something other than "zip" or "city" maybe you should try that again?')

        #   if the user entered a zip code and a city and state which match the size and character
        #   requirements then we can use that data to connect to the web service using  a
        #   weatherMapConnector object which we will call connect
        if start:
            connect = weatherMapConnector(userInputList, infoType)


if __name__ == '__main__':
    main()

```

    Welcome to the Aritzi Silva Weather Service, bringing you the most up to date weather patterns and futurecasts occurring in your area!
    
    If you would like to look up weather data by zip code, please enter the word 'zip'
    
    If you would like to look up weather data by city and state please enter the word 'city' to quit enter 'q'. 
    
    zip
    Please enter a valid zip code in the United States 
    34747
    The current weather in  Kissimmee :
    
    
    Clear ,  clear sky 
    
    The temperature is currently  70.65  and  83 % humidity
    
    The wind is blowing,  W  at  5.19  MPH
    
    Over the next couple days you can expect the forecast to look like: 
    
    
         DATE                         WEATHER            DEGREES       HUMIDITY  
    2021-04-16 06:00:00          scattered clouds          69.26          86 %
    
    2021-04-16 09:00:00          clear sky          67.33          92 %
    
    2021-04-16 12:00:00          scattered clouds          67.93          92 %
    
    2021-04-16 15:00:00          light rain          77.61          67 %
    
    2021-04-16 18:00:00          light rain          79.86          64 %
    
    2021-04-16 21:00:00          light rain          79.83          65 %
    
    2021-04-17 00:00:00          light rain          72.48          83 %
    
    2021-04-17 03:00:00          light rain          68.99          90 %
    
    2021-04-17 06:00:00          overcast clouds          67.42          92 %
    
    2021-04-17 09:00:00          overcast clouds          66.58          95 %
    
    2021-04-17 12:00:00          overcast clouds          68.31          92 %
    
    2021-04-17 15:00:00          overcast clouds          78.98          64 %
    
    2021-04-17 18:00:00          broken clouds          88.11          37 %
    
    2021-04-17 21:00:00          scattered clouds          86.36          38 %
    
    2021-04-18 00:00:00          few clouds          76.42          62 %
    
    2021-04-18 03:00:00          scattered clouds          71.71          79 %
    
    2021-04-18 06:00:00          scattered clouds          69.01          88 %
    
    2021-04-18 09:00:00          overcast clouds          67.93          92 %
    
    2021-04-18 12:00:00          broken clouds          69.44          88 %
    
    2021-04-18 15:00:00          scattered clouds          81.57          57 %
    
    2021-04-18 18:00:00          scattered clouds          88.34          42 %
    
    2021-04-18 21:00:00          overcast clouds          85.01          51 %
    
    2021-04-19 00:00:00          overcast clouds          77.67          69 %
    
    2021-04-19 03:00:00          light rain          72.5          85 %
    
    2021-04-19 06:00:00          light rain          69.35          92 %
    
    2021-04-19 09:00:00          light rain          69.33          92 %
    
    2021-04-19 12:00:00          light rain          68.18          92 %
    
    2021-04-19 15:00:00          light rain          66.4          85 %
    
    2021-04-19 18:00:00          light rain          66.52          83 %
    
    2021-04-19 21:00:00          light rain          68.31          81 %
    
    2021-04-20 00:00:00          light rain          66.78          87 %
    
    2021-04-20 03:00:00          light rain          64.65          92 %
    
    2021-04-20 06:00:00          light rain          64.2          88 %
    
    2021-04-20 09:00:00          light rain          64.85          91 %
    
    2021-04-20 12:00:00          light rain          66.49          95 %
    
    2021-04-20 15:00:00          light rain          74.64          82 %
    
    2021-04-20 18:00:00          moderate rain          79.47          71 %
    
    2021-04-20 21:00:00          moderate rain          74.14          84 %
    
    2021-04-21 00:00:00          moderate rain          70.99          94 %
    
    2021-04-21 03:00:00          moderate rain          68.49          93 %
    
    If you would like to look up weather data by zip code, please enter the word 'zip'
    
    If you would like to look up weather data by city and state please enter the word 'city' to quit enter 'q'. 
    
    10956
    Uh oh! Looks like you entered something other than "zip" or "city" maybe you should try that again?
    If you would like to look up weather data by zip code, please enter the word 'zip'
    
    If you would like to look up weather data by city and state please enter the word 'city' to quit enter 'q'. 
    
    y
    Uh oh! Looks like you entered something other than "zip" or "city" maybe you should try that again?
    If you would like to look up weather data by zip code, please enter the word 'zip'
    
    If you would like to look up weather data by city and state please enter the word 'city' to quit enter 'q'. 
    
    city
    Please enter a valid city in the United States 
    new city
    What state is that city in? Please use the 2 letter format: NY, CA, FL, etc. 
    NY
    Uh oh! Looks like you entered something in the wrong format! Remember, city should be a valid city name in the US made up of all letters and state should be the two letter state code, like NY for New York or FL for Florida
    If you would like to look up weather data by zip code, please enter the word 'zip'
    
    If you would like to look up weather data by city and state please enter the word 'city' to quit enter 'q'. 
    
    


```python

```
