package all

// Go 语言按空格分割字符串：
// arr := strings.Fields(s)

// Go 语言按字符串分割字符串语法：
// arr := strings.Split(s,sep)

// 拼接字符串切片：
// s := strings.Join(arr, sep)

// Go 语言按字符分割字符串语法：
// arr := strings.FieldsFunc(s,f func(rune) bool)

// 判断 字符串str 是否拥有该前缀
// strings.HasPrefix(str, prefix)

// 判断 字符串str 是否拥有该后缀
// strings.HasSuffix(str, suffix)

// 判断 字符串str 是否拥包含了该字符串
// strings.Contains(str, substr)

// 转换成小写
// nstr := strings.ToLower(str)

// 转换成大写
// nstr := strings.ToUpper(str)

// 是字母
// unicode.IsLetter(rune) bool

// 小写字母
// unicode.IsLower(rune) bool

// 大写字母
// unicode.IsUpper(rune) bool

// 数字
// unicode.IsDigit(rune) bool

// Manacher 算法
// 最长回文子串  https://leetcode.cn/problems/longest-palindromic-substring/
func longestPalindrome(s string) string {
	start, end := 0, -1
	t := "#"
	for _, v := range s {
		t += string(v) + "#"
	}
	s, n := t, len(t)

	expand := func(l, r int) int {
		for ; l >= 0 && r < n && s[l] == s[r]; l, r = l-1, r+1 {
		}
		return (r - l - 2) / 2
	}

	r, center := -1, -1  // 最长右边界与对应的中心
	rs := make([]int, n) // 记录每个字符对应的最长半径
	for i := range s {
		curr := 0 // 字符s[i]的最长半径
		if r > i {
			li := 2*center - i // 字符s[i]关于center的对称点
			mr := min(r-i, rs[li])
			curr = expand(i-mr, i+mr)
		} else {
			curr = expand(i, i)
		}
		rs[i] = curr
		if i+curr > r { // 更新最长右边界
			r, center = i+curr, i
		}
		if curr*2+1 > end-start+1 { // 更新答案
			start, end = i-curr, i+curr
		}
	}

	ans := ""
	for i := start; i <= end; i++ {
		if s[i] != '#' {
			ans += string(s[i])
		}
	}
	return ans
}

// Manacher 算法例题：
// 回文子字符串的个数 https://leetcode.cn/problems/a7VOhD/
