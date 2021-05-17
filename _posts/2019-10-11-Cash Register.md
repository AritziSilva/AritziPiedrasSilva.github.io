```python
"""
    File: Assignment 11.1. py
    Author: Aritzi Piedras-Silva
    Date: 11/10/2019
    Course: DSC 510 - Introduction to programing
    Desc: This program will run as a cash register program. User will be prompted to enter
    the price of items that they want to purchase when done purchasing the user will enter 'done'.
    This program will print the total amount of items purchased as well as the final price.
"""


import locale
# sets the locale
locale.setlocale(locale.LC_ALL, 'en_US')


class CashRegister:
    # constructor
    def __init__(self):
        self.total = 0
        self.items = 0

    # this will add the items in the cart
    def additem(self, price):
        self.items += 1
        try:
            price = float(price)
            self.total += price
        except:
            return

    # Getter method
    def getCount(self):
        return self.items

    # Getter method
    def getTotal(self):
        return locale.currency(self.total)


# this will ask the user for the price of item in dollars and cents.
def shopping():
    while True:
        try:
            costperitem = input('Cost of Item: $')
            if costperitem.lower() == 'done':
                return -1
            else:
                costperitem = float(costperitem)
                assert costperitem > 0.00
                return costperitem
        # this will pop if the user uses cost of item at 0.00
        except AssertionError:
            print('Sorry! Price must be greater than $0.00')


# final function
def main():
    # welcomes the user and explains the program and how to print the final receipt
    welcome = 'Welcome to my cash register! \n' \
              'In this program you will enter the price of each one of your items in dollar and cents. \n' \
              'To get your final receipt enter \'done\''
    finalregister = CashRegister()
    print(welcome)
    classregister = True
    while classregister:
        finalprices = shopping()
        if finalprices > 0:
            finalregister.additem(finalprices)
        else:
            classregister = False
    # prints the total of items and total cost of all items in US currency
    print('~' * 20 + '\nTotal amount of items in your cart:')
    print(finalregister.getCount())
    print('Your final price:')
    print(finalregister.getTotal())
    print('~' * 20)
    print('\nThank you for shopping with us! Come again')
    print('\n' + '~' * 20)


main()






```

    Welcome to my cash register! 
    In this program you will enter the price of each one of your items in dollar and cents. 
    To get your final receipt enter 'done'
    Cost of Item: $8.99
    Cost of Item: $5.10
    Cost of Item: $6.78
    Cost of Item: $done
    ~~~~~~~~~~~~~~~~~~~~
    Total amount of items in your cart:
    3
    Your final price:
    $20.87
    ~~~~~~~~~~~~~~~~~~~~
    
    Thank you for shopping with us! Come again
    
    ~~~~~~~~~~~~~~~~~~~~
    


```python

```
