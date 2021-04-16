```python
"""
    File: Assignment 10.1. py
    Author: Aritzi Piedras-Silva
    Date: 11/3/2019
    Course: DSC 510 - Introduction to programing
    Desc: This program will run different Chuck Norris jokes from the provided API.
"""

import requests

yes_prompt = ['y', 'yes', 'yasss', 'yup', 'yuh', 'yup']      # accepted yes' responses
no_prompt = ['n', 'no', 'nah', 'nope', 'nuh-uh']            # accepted no's responses
chuck_norris = "https://api.chucknorris.io/jokes/random"   # chuck norris url


# this will be used as the welcome message
def welcome_block():
    # welcome message string
    welcome = 'Welcome to Chuck Norris joke program! hope you emptied your bladder. Lets get started'
    print(welcome)


# this function calls the API
def chuck_jokes():
    try:
        joke = requests.get(chuck_norris)
        if joke.status_code != 200:
            print('Hmm... seems like we have been roundhouse kicked.')
            exit(joke.status_code)
        else:
            print('Hold on to your pants.... its coming....')
            # here the joke gets printed and is separated by ~
            print('~' * 20 + '\n' + joke.json()["value"] + '\n' + '~' * 20 + '\n')
    except:
        exit('Yikes ..I\'m trying i swear!')


# this function will be the good bye message of the program
def thank_you():
    bye = 'Hope you enjoyed these jokes! I know I did, come back soon!'
    separation_line = '~' * 10
    print(separation_line + '\n' + bye)


# in this function all other functions come together and get executed. the heart of the program
def main():

    welcome_block()
    user_answer = ''
    while not user_answer.lower() in (yes_prompt + no_prompt):
        print('Are you ready for the best Chuck Norris jokes?')
        # this gives the user option of reading the joke or not
        user_answer = input('If you want to read the jokes enter \'yasss\' or \'y\'.' 
                            'If you want to be bored then enter \'nah\' or \'n\' to exit:')
        if user_answer.lower() in no_prompt:
            break
        elif not user_answer.lower() in (yes_prompt + no_prompt):
            continue
        else:
            # this displays the joke by calling the joke block
            chuck_jokes()
        # this asks the user if they want to read more or not
        user_answer = ''
    # calls the good bye message
    thank_you()


main()


```

    Welcome to Chuck Norris joke program! hope you emptied your bladder. Lets get started
    Are you ready for the best Chuck Norris jokes?
    If you want to read the jokes enter 'yasss' or 'y'.If you want to be bored then enter 'nah' or 'n' to exit:y
    Hold on to your pants.... its coming....
    ~~~~~~~~~~~~~~~~~~~~
    Chuck Norris once squeezed a m&m so hard that it became a skittle.
    ~~~~~~~~~~~~~~~~~~~~
    
    Are you ready for the best Chuck Norris jokes?
    If you want to read the jokes enter 'yasss' or 'y'.If you want to be bored then enter 'nah' or 'n' to exit:yasss
    Hold on to your pants.... its coming....
    ~~~~~~~~~~~~~~~~~~~~
    Gordon Ramsay features Chuck Norris' Toaster Strudel recipe in all of his restaurants.
    ~~~~~~~~~~~~~~~~~~~~
    
    Are you ready for the best Chuck Norris jokes?
    


```python

```
