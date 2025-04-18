---
layout: post
title: while문 사용 과제
date: 2025-03-10 23:18 +0800
last_modified_at: 2025-03-10 23:20 +0800
tags: [python, assn]
toc:  true
---
**while문** 사용하여 사칙계산기 만들기
{: .message }

과제 : 
> while 문과 if문을 같이 사용하여 간단한 계산기 프로그램을 만들어 보세요.
> 사용자로부터 두 개의 숫자와 연산자(+, -, *, /)를 입력받아 
> 계산 결과를 출력하는 프로그램을 작성하세요.
> while 문을 사용하여 계산을 반복하도록 구현하세요.
> 사용자가 'q'를 입력하면 프로그램을 종료하도록 하세요.
> 0으로 나누는 경우 "0으로 나눌 수 없습니다."를 출력하고 다시 입력을 받으세요.
> 잘못된 연산자가 입력되면 "잘못된 연산자입니다."를 출력하고 다시 입력을 받으세요.

{% highlight js %}
while True:
    num1_input = input("첫번째 숫자를 입력 (종료: q): ")
    if num1_input == "q":
        break

    num2_input = input("두번째 숫자를 입력 (종료: q): ")
    if num2_input == "q":
        break

    ope = input("연산자 입력 (+, -, *, /)(종료: q): ")
    if ope == "q":
        break
    try:
        # 실행할 코드
        num1 = int(num1_input)
        num2 = int(num2_input)
    except ValueError:  #예외 발생 - 숫자가 아닌 값이 들어오면 에러 처리
        print("숫자를 입력하세요!")
        continue

    if ope == "+":
        print(f"더한 값: {num1 + num2}")
    elif ope == "-":
        print(f"뺀 값: {num1 - num2}")
    elif ope == "*":
        print(f"곱한 값: {num1 * num2}")
    elif ope == "/":
        if num2 == 0:
            print("0으로 나눌 수 없습니다.")
        else:
            print(f"나눈 값: {num1 / num2}")
    else:
        print("잘못된 연산자입니다. 다시 입력하세요.")

{% endhighlight %}

### 출력 결과
{% highlight text %}
첫번째 숫자를 입력 (종료: q): 50
두번째 숫자를 입력 (종료: q): 6
연산자 입력 (+, -, *, /)(종료: q): *
곱한 값: 300
첫번째 숫자를 입력 (종료: q): 50
두번째 숫자를 입력 (종료: q): 0
연산자 입력 (+, -, *, /)(종료: q): /
0으로 나눌 수 없습니다.
첫번째 숫자를 입력 (종료: q): 50
두번째 숫자를 입력 (종료: q): d
연산자 입력 (+, -, *, /)(종료: q): 
숫자를 입력하세요!
첫번째 숫자를 입력 (종료: q): 50
두번째 숫자를 입력 (종료: q): 6
연산자 입력 (+, -, *, /)(종료: q): 1
잘못된 연산자입니다. 다시 입력하세요.
첫번째 숫자를 입력 (종료: q): q
{% endhighlight %}

