def solution(s):
    # 나눌 숫자가 문자열의 반이상이 안되기때문에 문자열의 길이를 반으로 나눈값까지 split
    split_lens = int(len(s)/2)
    rs = []
    resultlen = []

    for i in range(1, split_lens + 1):
        strlst = []
        correctlens = []
        l = int(len(s)/i)
        # 몫과 나머지값
        # l:몫 p:나머지
        l , p = divmod(len(s), i)

        for j in range(l):
            if i==1:
                strlst.append(s[j])
            else:
                strlst.append(s[j*i:(j+1)*i])

        if p!=0:
            strlst.append(s[-p:])

        print(strlst)
        cnt = 0
        point = 0
        for z in range(len(strlst)):
            if strlst[z-1] == strlst[z]:
                cnt += 1
                if cnt == 1:
                    point = len(correctlens)-1
                    if point == -1:
                        correctlens.append(str(cnt + 1) + strlst[z])

                    correctlens[point] = str(cnt + 1) + strlst[z]
                else:
                    correctlens[point] = str(cnt+1) + strlst[z]
            elif z == 0:
                correctlens.append(strlst[z])
            else :
                cnt = 0
                point = 0
                correctlens.append(strlst[z])
            print(correctlens)
        join_str = ''.join(correctlens)
        resultlen.append(join_str)

    for x in range(len(resultlen)):
        rs.append(len(resultlen[x]))

    answer = min(rs)
    return answer


str_ = "abcabcdede"

print(solution(str_))