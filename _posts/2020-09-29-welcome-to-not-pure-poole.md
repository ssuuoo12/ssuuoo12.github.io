---
layout: post
title: LSTM 과제(딥러닝)
date: 2025-03-21 23:18 +0800
last_modified_at: 2025-03-21 01:08:25 +0800
tags: [jekyll theme, jekyll, tutorial]
toc:  true
---
LSTM  **딥러닝** 시계열 데이터 과제
{: .message }

<!-- First, do you notice the TOC on the right side? Try to scroll down to read this post, you'll find that the TOC is always sticky in the viewport.

Cum sociis natoque penatibus et magnis <a href="#">dis parturient montes</a>, nascetur ridiculus mus. *Aenean eu leo quam.* Pellentesque ornare sem lacinia quam venenatis vestibulum. Sed posuere consectetur est at lobortis. Cras mattis consectetur purus sit amet fermentum.

> Curabitur blandit tempus porttitor. Nullam quis risus eget urna mollis ornare vel eu leo. Nullam id dolor id nibh ultricies vehicula ut id elit.

Etiam porta **sem malesuada magna** mollis euismod. Cras mattis consectetur purus sit amet fermentum. Aenean lacinia bibendum nulla sed consectetur. -->

<!-- ## Inline HTML elements

HTML defines a long list of available inline tags, a complete list of which can be found on the [Mozilla Developer Network](https://developer.mozilla.org/en-US/docs/Web/HTML/Element).

- **To bold text**, use `<strong>`.
- *To italicize text*, use `<em>`.
- <mark>To highlight</mark>, use `<mark>`.
- Abbreviations, like <abbr title="HyperText Markup Langage">HTML</abbr> should use `<abbr>`, with an optional `title` attribute for the full phrase.
- Citations, like <cite>&mdash; Mark Otto</cite>, should use `<cite>`.
- <del>Deleted</del> text should use `<del>` and <ins>inserted</ins> text should use `<ins>`.
- Superscript <sup>text</sup> uses `<sup>` and subscript <sub>text</sub> uses `<sub>`.

Most of these elements are styled by browsers with few modifications on our part. -->

<!-- ## Footnotes

Footnotes are supported as part of the Markdown syntax. Here's one in action. Clicking this number[^fn-sample_footnote] will lead you to a footnote. The syntax looks like:

{% highlight text %}
Clicking this number[^fn-sample_footnote]
{% endhighlight %}

Each footnote needs the `^fn-` prefix and a unique ID to be referenced for the footnoted content. The syntax for that list looks something like this:

{% highlight text %}
[^fn-sample_footnote]: Handy! Now click the return link to go back.
{% endhighlight %}

You can place the footnoted content wherever you like. Markdown parsers should properly place it at the bottom of the post.
 -->


## Code

<!-- Inline code is available with the `<code>` element. Snippets of multiple lines of code are supported through Rouge. Longer lines will automatically scroll horizontally when needed. You may also use code fencing (triple backticks) for rendering code. -->

<!-- {% highlight js %}
// Example can be run directly in your JavaScript console

// Create a function that takes two arguments and returns the sum of those arguments
var adder = new Function("a", "b", "return a + b");

// Call the function
adder(2, 6);
// > 8
{% endhighlight %}

You may also optionally show code snippets with line numbers. Add `linenos` to the Rouge tags.

{% highlight js linenos %}
// Example can be run directly in your JavaScript console

// Create a function that takes two arguments and returns the sum of those arguments
var adder = new Function("a", "b", "return a + b");

// Call the function
adder(2, 6);
// > 8
{% endhighlight %}

Aenean lacinia bibendum nulla sed consectetur. Etiam porta sem malesuada magna mollis euismod. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa. -->
### seoul_pm10 미세먼지 예측 과제
코랩 dataset 폴더에 있는 seoul_pm10.csv 파일을 다운 받고, 강남구 pm10에 대한 실제값과 예측값을 시각화하여 비교하세요.

1. 데이터 로드 : "dataset/seoul_pm10.csv"   ==> encoding='cp949' 유의
2. 날짜 변환 및 결측치 처리 : to_datetime
3. 서울 지역별 원핫 인코딩
4. LSTM 모델에 적합한 시퀀스 데이터셋 함수 생성
5. 데이터셋 분리 (학습 데이터, 테스트 데이터)
6. LSTM 모델 생성
7. 모델 컴파일 및 학습
8. 예측 결과 시각화

{% highlight js %}


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

// 1. 데이터 수집
file_path = "./sample_data/seoul_pm10.csv"  # 데이터 파일 경로
df = pd.read_csv(file_path, encoding='cp949')

// df.head()

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

// 결측치 처리
df['pm10'] = df['pm10'].fillna(df['pm10'].mean())
df['pm2.5'] = df['pm2.5'].fillna(df['pm2.5'].mean())

// 지역 원-핫 인코딩
df_encoded = pd.get_dummies(df, columns=['area'], prefix='area')

// 강남구 데이터만 필터링
df_gangnam = df_encoded[df_encoded['area_강남구'] == 1].copy()

// 2. 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_gangnam[['pm10']])

def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 60  # 과거 60일 데이터로 예측
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM 입력 형태

// 3. 모델 구성
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

// 4. 모델 학습
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

// 5. 예측 및 시각화
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

// 실제 값 복원
actual_pm10 = scaler.inverse_transform(y_test.reshape(-1, 1))

// 시각화
plt.figure(figsize=(14, 5))
plt.plot(actual_pm10, label="Actual Pm10", color='blue')
plt.plot(predictions, label="Predicted Pm10", color='red')
plt.title(f'Gangnam Pm10 Predicted')
plt.xlabel('Days')
plt.ylabel('Pm10')
plt.legend()
plt.show()


{% endhighlight %}


### Lists

<!-- Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aenean lacinia bibendum nulla sed consectetur. Etiam porta sem malesuada magna mollis euismod. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit amet risus. -->

<!-- - Praesent commodo cursus magna, vel scelerisque nisl consectetur et.
- Donec id elit non mi porta gravida at eget metus.
- Nulla vitae elit libero, a pharetra augue. -->

1. 데이터 수집
- pd.read_csv(file_path, encoding='cp949'): CSV 파일을 읽어 데이터프레임 생성.
- df['date'] = pd.to_datetime(df['date']): 날짜 열을 datetime 형식으로 변환.
- df.set_index('date', inplace=True): 날짜를 데이터프레임 인덱스로 설정.
- df['pm10'].fillna(df['pm10'].mean()): PM10 결측치를 평균으로 채움.
- df['pm2.5'].fillna(df['pm2.5'].mean()): PM2.5 결측치를 평균으로 채움.
- pd.get_dummies(df, columns=['area']): 지역 열을 원-핫 인코딩.
- df_gangnam = df_encoded[df_encoded['area_강남구'] == 1]: 강남구 데이터만 필터링.
2. 데이터 전처리
- scaler = MinMaxScaler(feature_range=(0, 1)): 0~1로 스케일링하는 객체 생성.
- scaled_data = scaler.fit_transform(df_gangnam[['pm10']]): PM10 데이터를 스케일링.
- def create_dataset(dataset, look_back=60): 60일 데이터를 사용해 X, y 생성 함수 정의.
- X, y = create_dataset(scaled_data, look_back): 스케일링된 데이터로 X, y 생성.
- X = np.reshape(X, (X.shape[0], X.shape[1], 1)): LSTM 입력 형태로 데이터 재구성.
3. 모델 구성
- model = Sequential(): 순차적 모델 초기화.
- LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)): 첫 번째 LSTM 레이어.
- LSTM(50): 두 번째 LSTM 레이어.
- Dense(1): 출력 레이어.
- model.compile(optimizer='adam', loss='mean_squared_error'): 모델 컴파일.

<!-- 1. Vestibulum id ligula porta felis euismod semper.
2. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.
3. Maecenas sed diam eget risus varius blandit sit amet non magna. -->

<!-- Cras mattis consectetur purus sit amet fermentum. Sed posuere consectetur est at lobortis.

<dl>
  <dt>HyperText Markup Language (HTML)</dt>
  <dd>The language used to describe and define the content of a Web page</dd>

  <dt>Cascading Style Sheets (CSS)</dt>
  <dd>Used to describe the appearance of Web content</dd>

  <dt>JavaScript (JS)</dt>
  <dd>The programming language used to build advanced Web sites and applications</dd>
</dl>

Integer posuere erat a ante venenatis dapibus posuere velit aliquet. Morbi leo risus, porta ac consectetur ac, vestibulum at eros. Nullam quis risus eget urna mollis ornare vel eu leo.

### Images

Quisque consequat sapien eget quam rhoncus, sit amet laoreet diam tempus. Aliquam aliquam metus erat, a pulvinar turpis suscipit at.

![placeholder](http://placehold.it/800x400 "Large example image")
![placeholder](http://placehold.it/400x200 "Medium example image")
![placeholder](http://placehold.it/200x200 "Small example image")

Align to the center by adding `class="align-center"`:

![placeholder](http://placehold.it/400x200 "Medium example image"){: .align-center} -->
<!-- 
### Tables

Aenean lacinia bibendum nulla sed consectetur. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Upvotes</th>
      <th>Downvotes</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>Totals</td>
      <td>21</td>
      <td>23</td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>Alice</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Charlie</td>
      <td>7</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

Nullam id dolor id nibh ultricies vehicula ut id elit. Sed posuere consectetur est at lobortis. Nullam quis risus eget urna mollis ornare vel eu leo.

-----

Want to see something else added? <a href="https://github.com/vszhub/not-pure-poole/issues/new">Open an issue.</a>

[^fn-sample_footnote]: Handy! Now click the return link to go back. -->
