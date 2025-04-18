---
layout: post
title: 카카오 지도 api not defined 에러 해결법
date: 2025-04-05 23:18 +0800
last_modified_at: 2025-04-06 23:20 +0800
tags: [error, api]
toc:  true
---
 **카카오 지도 api** 에러 해결법법
{: .message }

{% highlight text %}
API 키 발급 방법을 알아보려고 일단 구글링을 해봤더니 관련 블로그 글이 꽤 많이 나오더라고요.
블로그에 나온 과정을 따라가면서 진행했는데, 처음엔 API 키만 발급받으면 바로 사용할 수 있을 줄 알았어요. 
하지만 막상 해보니 계속 "kakao is not defined"라는 에러가 뜨는 거예요. 여기서 좀 헤매면서 
시간이 꽤 걸렸습니다.
에러 메시지를 추가해서 다시 검색해보니, 알고 보니 카카오맵을 활성화해야 제대로 사용할 수 있더라고요. 
이런 실수를 할 줄이야… 
아무래도 블로그 내용만 너무 맹신했던 것 같아요. 
그래서 카카오 개발자 사이트로 돌아가서 카카오맵을 활성화하고, 앱 권한도 신청했어요. 그랬더니 
10분 정도 기다린 후에 지도가 정상적으로 출력되더라고요. 
나중에 정리해보니, 정확한 과정은 ‘내 애플리케이션 > 제품 설정 > 카카오맵 > 활성화 설정 ON’으로 설정하고, 
‘내 애플리케이션 > 앱 설정 > 앱 권한 신청’까지 완료해야 했던 거였어요. 처음엔 좀 당황했지만, 
결국 해결하고 나니 한결 마음이 놓였습니다.
{% endhighlight %}


![지도 에러](/notd.png "에러 사진")
- 콘솔창에는 "kakao is not defined"라고 계속 떴다.
![지도 에러2](/지도오류.png "에러 사진2")
- Network창에는 "401"라고 권한이 계속 없다고 떴다.

![설정](/지도권한.png "설정")
![설정2](/활성화.png "설정2")
- 권한과 활성화를 시켜주니까 해결이 되었다.

