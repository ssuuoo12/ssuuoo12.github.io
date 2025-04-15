---
layout: post
title: 백준 알고리즘 1874 스택 수열 with java
date: 2025-04-15 23:18 +0800
last_modified_at: 2025-04-15 23:20 +0800
tags: [Algorithm, java, stack]
toc:  true
---
  백준 알고리즘 1874번 **스택 수열** 
{: .message }

[문제 보러가기기](https://www.acmicpc.net/problem/1874).

{% highlight text %}
예제 입력이 [5, 1, 2, 5, 3, 4]일 때 진행 과정은:
첫 번째 숫자 5를 처리할 때:
current = 1부터 시작
current ≤ target(5)이므로 1→2→3→4→5를 push
이 과정 후 current = 6이 됨
5를 pop

두 번째 숫자 1을 처리할 때:
current = 6, 스택에는 [1,2,3,4]가 남아 있음
current > target(1)이므로 push 수행 안함
스택의 맨 위는 4이고 target은 1이므로 불일치
여기서 "NO"가 출력됨

왜 "NO"가 출력되는지
스택의 특성상 나중에 들어간 요소가 먼저 나오게 됩니다(LIFO). 이 문제에서는 1부터 n까지 오름차순으로만 push할 수 있습니다.
예제 [5, 1, 2, 5, 3, 4]에서:

5를 처리한 후 스택에는 [1,2,3,4]가 남아있음
다음으로 1을 꺼내야 하는데, 스택의 맨 위는 4임
스택의 맨 위에서부터 요소를 꺼내야 하므로 1에 접근할 수 없음
따라서 이 수열은 스택으로 만들 수 없음

!stack.isEmpty(): 스택이 비어있지 않은지 확인
비어있으면 peek()나 pop()을 호출할 때 예외가 발생하므로 먼저 검사

stack.peek() == target: 스택의 맨 위 요소가 현재 찾는 target과 같은지 확인
peek()는 스택의 맨 위 요소를 제거하지 않고 값만 반환

두 조건이 모두 참일 시에만 target 숫자를 pop할 수 있습니다. 그렇지 않으면 현재 수열을 만들 수 없으므로 "NO"를 출력
이 예제에서는 두 번째 숫자(1)를 처리할 때 스택의 맨 위가 4이므로 1을 얻을 수 없어 불가능한 경우가 되어 "NO"를 출력
{% endhighlight %}

### 풀이
{% highlight js %}
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Stack;

public class Stack_Sequence_1874 {
	
	 public static void main(String[] args) throws IOException {
		 
	        // BufferedReader: Scanner보다 빠른 입력을 위한 클래스
	        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
	        // StringBuilder: 문자열을 효율적으로 연결하기 위한 클래스
	        StringBuilder sb = new StringBuilder();
	        
	        // 첫 줄에서 정수 n 입력받기(수열의 길이)
	        int n = Integer.parseInt(br.readLine());
	        
	        // 스택 생성
	        Stack<Integer> stack = new Stack<>();
	        int current = 1; // 스택에 넣을 다음 숫자(오름차순으로 push)
	        
	        // 수열 생성 가능 여부 체크 변수
	        boolean isPossible = true;
	        
	        // n개의 숫자에 대해 반복
	        for (int i = 0; i < n; i++) {
	            // 목표 숫자 입력받기(현재 만들어야 할 수열의 숫자)
	            int target = Integer.parseInt(br.readLine());
	            
	            // 현재 숫자가 목표 숫자보다 작거나 같을 때까지 스택에 push
	            while (current <= target) {
	                stack.push(current++); // 스택에 현재 숫자 넣고 current 1 증가
	                sb.append("+\n");      // push 연산 기록(+)
	            }
	            
	            // 스택의 맨 위 숫자가 목표 숫자와 같은지 확인
	            if (!stack.isEmpty() && stack.peek() == target) {
	                // peek(): 스택의 맨 위 요소를 확인(제거하지 않음)
	                stack.pop(); // 스택에서 숫자 꺼내기
	                sb.append("-\n"); // pop 연산 기록(-)
	            } else {
	                // 불가능한 경우: 스택 맨 위 숫자가 목표 숫자와 다름
	                isPossible = false;
	                break;
	            }
	        }
	        
	        // 결과 출력
	        if (isPossible) {
	            System.out.print(sb); // 가능한 경우 push/pop 연산 순서 출력
	        } else {
	            System.out.println("NO"); // 불가능한 경우 NO 출력
	        }
	    }
}

{% endhighlight %}

