import numpy as np

days = ["Mo","Tu","We", "Th", "Fr","Sa", "Su"]
months = ["Jan","Feb","Mar","Apr","May","June","July","Aug","Sept","Oct","Nov","Dec"]
# function calculates wheather the year is a leap or not
def leap_or_not(year):
    val4 = year/4
    val100 = year/100
    val400 = year/400
    val3200 = year/3200
    
    if val4 - int(val4) == 0:
        if val100 - int(val100) == 0:
            if val400 - int(val400) == 0:
                if val3200 - int(val3200) != 0:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 1
    else:
        return 0
# calculates the number of days in a month   
def No_month_days(month, leap):
    if month == 1:
        if leap == 1:
            return 30
        else:
            return 29
    else:
        val = month/2
        val2 = int(val)
        val3 = val - val2
        if val3 == 0:
            return 32
        else:
            return 31
# formula for calculating the day of the starting year        
def Zellers_congruence(q,year):
    k = k_val(year-1)
    m = m_val(1)
    j = j_val(year)
    h = (q + (int((13*(m+1))/5)) + k + int(k/4) + int(j/4) - (2*j)) % 7
    d = ((h+5)% 7)+1
    return d

def m_val(month):
    if month <= 2:
        num_month = month + 12
        return num_month
    else:
        return month

def k_val(year):
    k = year % 100
    return k
   
def j_val(year):
    j = np.floor(year/100)
    return j


def start_pos(year):
    q = 1
    val = int(round(Zellers_congruence(q, year)))
    print(val)
    return val -1 
# this function finds the start day of year (1st jan) then creates calender for the rest of the year.      
def calender(year):
    leap = leap_or_not(year)
    start_pos1 = start_pos(year)
    print(start_pos1)
    for month in range(0,12):
        print("{0} {1}".format(months[month], year).center(20, " "))
        print("".join(["{0:<3}".format(d) for d in days]))
        print("{0:<3}".format("")*start_pos1, end="")
        number_days = No_month_days(month, leap)
        for day in range(1, number_days):
            print("{0:<3}".format(day), end="")
            start_pos1 += 1
            if start_pos1 == 7:
                print()
                start_pos1 = 0
        print("\n")
        
yr=int(input("Enter the year of the calender required :")) 
calender(yr)

