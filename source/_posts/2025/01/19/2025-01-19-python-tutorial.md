---
title: Python Tutorial
date: 2025-01-19 21:35:10
tags: Python
---

## Python Version

3.10 3.11 3.12

```
# c/c++
/ %
c:/ == python: /
c:% == python: //

python:
+
-
*
/ //
```

## Function

```python
def hello(name):
		print("hello ", name)
```

## Data Type

```python
a:int = 1
b:float = 2.1
c:str = "this is a string"
print(f"a={a}")
print(f"b={b}")
print(f"c={c}")
# char 'a'
# string "a"
c1 = c + "abc"
print(f"c1={c1}")
```

## Class

```python
class Apple:
    def __init__(self, color, weight):
        self.color = color
        self.weight = weight

    def double_weight(self):
        self.weight = self.weight * 2

    def __str__(self):
        return f"color: {self.color}, weight: {self.weight}"


def main():
    red_apple = Apple("red", 100)
    print(f"color {red_apple.color}")  # red
    print(f"weight {red_apple.weight}")  # 100

    red_apple.double_weight()
    print(f"color {red_apple.color}")  # red
    print(f"weight {red_apple.weight}")  # 200

    print(red_apple)

    pass


if __name__ == '__main__':
    main()

```



## Json Type

```json
[
  {
    "name": "mike",
    "age": 18
  }
]
```

## Dict

```python
def main():
  
    dict1 = {
        "name": "Mike",
        "age": 18,
      	# "apple": apple
    }

    print(dict1)
    print(dict1["age"])
    pass


if __name__ == '__main__':
    main()

```

## Tuple

```python
def main():
    tuple1 = ("mike", 2.0, 3)
    print(tuple1)
    print(tuple1[0:2])
    pass


if __name__ == '__main__':
    main()

```

## List && if elif else && for

```python
class Apple:
    def __init__(self, color, weight):
        self.color = color
        self.weight = weight

    def double_weight(self):
        self.weight = self.weight * 2

    def __str__(self):
        return f"color: {self.color}, weight: {self.weight}"


def main():
    apple = Apple('red', 5) # 实例化了一个对象，叫apple
    list1 = [1, 2, 3, 4, 5, "mike", 22.2, apple]
    print(list1)
    for i in list1:
        if i == "mike":
            print(i)
        elif i == 1:
            print(i)
        else:
            print("End")
    pass


if __name__ == '__main__':
    main()

```

```python
def factor(n):
    s = 1
    for i in range(1, n + 1):
        s *= i  # s = s * i
    return s


def main():
    n = factor(10)
    print(n)
    pass


if __name__ == '__main__':
    main()

```

## File

```python
def main():
    print("hello\nworld\n\nmike")
    with open("word.txt") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
    pass


if __name__ == '__main__':
    main()
```



```python
def bubble_sort(array):
    for i in range(len(array)):
        for j in range(len(array)):
            if array[i] < array[j]:
                array[i], array[j] = array[j], array[i]


def main():
    l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    bubble_sort(l)
    print(l)

    pass


if __name__ == '__main__':
    main()

```

