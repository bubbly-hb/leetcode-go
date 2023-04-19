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

# 函数与方法
```go
package main

import (
	"container/heap"
	"fmt"
)

// 接受一个值或者指针作为参数的函数必须接受一个指定类型的值（值对应值，指针对应指针）
// 而以值或指针为接收者的方法被调用时，接收者既能为值又能为指针，方法调用会自动解释为对应的类型
func main() {
	t := hp{}
	heap.Push(&t, 23)
	t.push(3)
	fmt.Println(t)
}

type hp []int

func (h hp) Len() int            { return len(h) }
func (h hp) Less(i, j int) bool  { return h[i] < h[j] }
func (h hp) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *hp) Push(x interface{}) { *h = append(*h, x.(int)) }
func (h *hp) Pop() interface{} {
	t := (*h)[h.Len()-1]
	(*h) = (*h)[:h.Len()-1]
	return t
}
func (h *hp) push(x int) { heap.Push(h, x) }
func (h *hp) pop() int   { return heap.Pop(h).(int) }
```

# int 与 float 混合运算
```go
package main

import "fmt"

func main() {
	a := 3
	fmt.Println(a / 10.0)          // 0
	fmt.Println(float64(a) / 10.0) // 0.3
}
```

# 奇数判断
& 1 > 0 比 & 1 == 1 要快

# 切片拷贝
预分配了空间的可以用copy，否则可以用append：
```go
copy(a, b)
a = append([]int(nil), b...)
```

# bfs队列问题
```go
a := []int{2}
for len(a) > 0 {
    t, a := a[0], a[1:]
    // ...
}
```
这样会把a当作局部变量，正确的写法如下：
```go
a := []int{2}
for len(a) > 0 {
    var t int
    t, a = a[0], a[1:]
    // ...
}
```
或者是：
```go
a := []int{2}
for len(a) > 0 {
    t := a[0]
    a = a[1:]
    // ...
}
```

# 最值枚举优化
求最大值时可以从大到小枚举，一旦符合直接break

# 函数内函数的递归每次运行结果不同
数组中的第K大的数字：
```go
func findKthLargest(nums []int, k int) int {
    dic := map[int]struct{}{}
    for _, v := range nums {
        dic[v] = struct{}{}
    }
    n := len(dic)
    a := make([]int, 0, n)
    for k, _ := range dic {
        a = append(a, k)
    }
    findPos := func(l, r int) int {
        // fmt.Println(l, r)
        randPos := rand.Intn(r - l + 1) + l
        a[l], a[randPos] = a[randPos], a[l]
        pos := l
        for i := l + 1; i <= r; i++ {
            if a[i] > a[l] {
                a[pos+1], a[i] = a[i], a[pos+1]
                pos++
            }
        }
        a[pos], a[l] = a[l], a[pos]
        fmt.Println(l, r, randPos, a, pos)
        return pos
    }
    var quickSelect func(int, int, int) int
    quickSelect = func(l, r, k int) int {
        if l == r {
            return a[l]
        }
        pos := findPos(l, r)
        if pos == k {
            return a[pos]
        } else if pos > k {
            return quickSelect(l, pos - 1, k)
        } else {
            return quickSelect(pos + 1, r, k) 
        }
    }
    return quickSelect(0, n - 1, k - 1)
}
```
上面代码中的quickSelect有问题，改为：
```go
quickSelect = func(l, r, k int) int {
    if l == r {
        return a[l]
    }
    pos := -1
    for pos != k {
        pos = findPos(l, r)
        if pos > k {
            r = pos - 1
        } else if pos < k {
            l = pos + 1
        }
    }
    return a[k]
}
```

# 枚举子集
若要枚举x的子集：
```go
for i := x; i > 0; i = (i - 1) & x {
    ans += dic[i]
}
ans += dic[0]  // 再对 i == 0 的情况做处理
```
直接像下面这么写，当x为0时会死循环，因为-1的二进制全为1，与0按位与后还是0：
```go
for i := x; i >= 0; i = (i - 1) & x {
    ans += dic[i]
}
```

# 哈希改数组常数优化(同时避免负数下标)
```go
func beautifulSubsets(nums []int, k int) (c int) {
    n := len(nums)
    // dic := map[int]int{}
    dic := make([]int, 1001 + k * 2)
    var dfs func(x int)
    dfs = func(x int) {
        if x == n {
            c++
            return
        }
        dfs(x+1)
        // if dic[nums[x] + k] > 0 || dic[nums[x] - k] > 0 {
        //     return
        // }
        // dic[nums[x]]++
        // dfs(x+1)
        // dic[nums[x]]--
        if dic[nums[x]] > 0 || dic[nums[x] + 2 * k] > 0 {
            return
        }
        dic[nums[x] + k]++
        dfs(x + 1)
        dic[nums[x] + k]--
    }
    dfs(0)
    return c - 1
}
```

# 函数体外声明变量
```go
var primes = []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
var toMask = [31]int{}
func init() {
    // do sth
}
```
而不是
```go
primes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
toMask := [31]int{}
func init() {
    // do sth
}
```

# rand
```go
rand.Seed(time.Now().UnixNano()) // 纳秒
```
如果每次调rand.Intn()前都调了rand.Seed(x)，每次的x相同的话，每次的rand.Intn()也是一样的。
两种解决方案：
1. 只调一次rand.Seed()：在全局初始化调用一次seed，每次调rand.Intn()前都不再调rand.Seed()。
2. 调多次rand.Seed(x)，但每次x保证不一样：每次使用纳秒级别的种子。强烈不推荐这种，因为高并发的情况下纳秒也可能重复。

# this is a test for dell2