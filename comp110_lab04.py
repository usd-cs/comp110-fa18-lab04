"""
Module: comp110_lab04

Modules with some functions for Lab 04 practice problems.
"""

import turtle

def sum_string_digits(my_str):
    """
    Sums up the digits in the input string and returns this sum.
    """

    return 0    # this is a placeholder. remove it.


def move_turtle(t, dir_str):
    """
    Uses the dir_string to move the turtle around the world.
    """

    for d in dir_str:
        if d == 'F':
            print("hi")    # this line is a placeholder, remove it.

        # need to complete this with some elifs and maybe an else...


def main():
    print("Input a turtle direction string:")
    dirs = input()

    turt = turtle.Turtle()
    move_turtle(turt, dirs)
    turtle.done()
