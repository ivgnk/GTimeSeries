'''
https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe
'''
import pandas as pd

def add_row1():
    s1 = pd.Series([5, 6, 7])
    s2 = pd.Series([7, 8, 9])

    df = pd.DataFrame([list(s1), list(s2)],  columns =  ["A", "B", "C"])
    print(df)
    df.loc[-1] = [2, 3, 4]
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index
    print(df)

def add_row2():
    df = pd.DataFrame(columns=['a', 'b'])
    for i in range(5):
        # df = pd.concat([df, pd.DataFrame([[i, i+2]], columns=df.columns)], ignore_index=True)
        df = pd.concat([df, pd.DataFrame([[i, i + 2]], columns=df.columns)])
    print(df)

def add_row3():
    # https://www.codecamp.ru/blog/pandas-add-row-to-dataframe/
    df = pd.DataFrame(columns=['a', 'b'])
    for i in range(5):
        df.loc[len(df.index)] = [i, i + 2]
        # df = pd.concat([df, pd.DataFrame([[i, i + 2]], columns=df.columns)])
    print(df)

if __name__ == "__main__":
    add_row3()
