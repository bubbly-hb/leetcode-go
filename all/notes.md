# [删除无效的括号](https://leetcode.cn/problems/remove-invalid-parentheses/)
+ 第一种做法是从左到右判断当前位删不删，不删的话加到cur里，然后哈希去重
```go
dfs = func(l, r, cl, cr, x int) {
    // fmt.Println(l, r, cl, cr, x, string(cur))
    if x == len(s) {
        //...
        return
    }
    if s[x] == '(' {
        //...
    } else if s[x] == ')' {
        //...
    } else {
        cur = append(cur, s[x])
        dfs(l, r, cl, cr, x + 1)
        cur = cur[:len(cur) - 1]   // 很关键
    }
}
```
+ 第二种做法是从左到右判断当前位删不删，删了的话把左右两边拼起来然后继续dfs, 在for里面去重

# [通配符匹配](https://leetcode.cn/problems/wildcard-matching/)
+ '?' 可以匹配任何单个字符。'*' 可以匹配任意字符串（包括空字符串）  
+ 下面注释部分可以替换成一句话
```go
for i, v := range s {
    for j, t := range p {
        if t == '?' || t == v {
            dp[i + 1][j + 1] = dp[i][j]
        } else if t == '*' {
            // for k := i + 1; k >= 0; k-- {
            //     dp[i+1][j+1] = dp[i+1][j+1] || dp[k][j]
            //     if dp[i+1][j+1] {
            //         break
            //     }
            // }
            dp[i + 1][j + 1] = dp[i + 1][j] || dp[i][j + 1]
        }
    }
}
```



