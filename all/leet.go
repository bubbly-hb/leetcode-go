package all

import (
	"container/heap"
	"container/list"
	"math"
	"math/bits"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// 统计特殊整数 数位dp
func countSpecialNumbers(n int) int {
	s := strconv.Itoa(n)
	ls := len(s)

	dp := make([][1 << 10]int, ls)
	for i := range dp {
		for j := range dp[i] {
			dp[i][j] = -1
		}
	}

	var dfs func(int, int, bool, bool) int
	dfs = func(x, mask int, isLimit, isNum bool) (ans int) {
		if x == ls {
			if isNum {
				return 1
			}
			return
		}

		if !isLimit && isNum {

			if dp[x][mask] >= 0 {
				return dp[x][mask]
			}
			defer func() { dp[x][mask] = ans }()
		}

		if !isNum {
			ans += dfs(x+1, mask, false, false)
		}

		d := 1
		if isNum {
			d = 0
		}

		up := 9
		if isLimit {
			up = int(s[x] - '0')
		}

		for ; d <= up; d++ {
			if mask>>d&1 == 0 {
				ans += dfs(x+1, mask|(1<<d), isLimit && d == up, true)
			}
		}
		return
	}
	return dfs(0, 0, true, false)
}

// 滑动窗口的最大值_优先队列
// var a []int

// type hp struct{ sort.IntSlice }

// func (h hp) Less(i, j int) bool  { return a[h.IntSlice[i]] > a[h.IntSlice[j]] }
// func (h *hp) Push(i interface{}) { h.IntSlice = append(h.IntSlice, i.(int)) }
// func (h *hp) Pop() interface{} {
// 	ls := len(h.IntSlice) - 1
// 	ans := h.IntSlice[ls]
// 	h.IntSlice = h.IntSlice[:ls]
// 	return ans
// }
// func maxSlidingWindow(nums []int, k int) []int {
// 	n := len(nums)
// 	a = nums
// 	q := &hp{make([]int, k)}
// 	for i := 0; i < k; i++ {
// 		q.IntSlice[i] = i
// 	}
// 	heap.Init(q)
// 	ans := make([]int, 1, n-k+1)
// 	ans[0] = a[q.IntSlice[0]]
// 	for i := k; i < n; i++ {
// 		heap.Push(q, i)
// 		for q.IntSlice[0] <= i-k {
// 			heap.Pop(q)
// 		}
// 		ans = append(ans, a[q.IntSlice[0]])
// 	}
// 	return ans
// }

// 滑动窗口最大值_单调队列
func maxSlidingWindow2(nums []int, k int) []int {
	n := len(nums)
	window := []int{}
	push := func(i int) {
		for len(window) > 0 && nums[window[len(window)-1]] <= nums[i] {
			window = window[:len(window)-1]
		}
		window = append(window, i)
	}
	for i := 0; i < k; i++ {
		push(i)
	}
	ans := make([]int, 1, n-k+1)
	ans[0] = nums[window[0]]
	for i := k; i < n; i++ {
		push(i)
		for window[0] <= i-k {
			window = window[1:]
		}
		ans = append(ans, nums[window[0]])
	}
	return ans
}

// 找到所有好字符串 数位dp + KMP算法
func findGoodStrings(n int, s1 string, s2 string, evil string) int {
	MOD := int(1e9 + 7)
	m := len(evil)
	dp := make([][51]int, n)
	for i := range dp {
		for j := range dp[i] {
			dp[i][j] = -1
		}
	}
	next := make([]int, m+1)
	for i := range next {
		next[i] = -1
	}
	l, r := 0, -1
	for l < m {
		if r == -1 || evil[l] == evil[r] {
			l++
			r++
			next[l] = r
		} else {
			r = next[r]
		}
	}
	var dfs func(int, int, bool, bool) int
	dfs = func(x, mask int, isUpLimit, isLowLimit bool) (ans int) {
		if mask == m {
			return 0
		}
		if x == n {
			return 1
		}

		if !isUpLimit && !isLowLimit {
			memo := &dp[x][mask]
			if *memo >= 0 {
				return *memo
			}
			defer func() { *memo = ans }()
		}

		d := 0
		if isLowLimit {
			d = int(s1[x] - 'a')
		}

		up := 25
		if isUpLimit {
			up = int(s2[x] - 'a')
		}

		for ; d <= up; d++ {
			ne := mask

			for ne != -1 && int(evil[ne]-'a') != d {
				ne = next[ne]
			}

			ans += dfs(x+1, ne+1, isUpLimit && d == up, isLowLimit && d == int(s1[x]-'a'))
			ans %= MOD

		}
		return
	}
	return dfs(0, 0, true, true)
}

// 长度最小的子数组 二分，SearchInts 或者用滑动窗口更快
func minSubArrayLen(target int, nums []int) int {
	n := len(nums)
	pre := make([]int, n)
	pre[0] = nums[0]
	for i := 1; i < n; i++ {
		pre[i] = pre[i-1] + nums[i]
	}
	if pre[n-1] < target {
		return 0
	}
	ans := n
	for i := 0; i < n; i++ {
		// l, r, pos := i, n - 1, i - 1
		// for l <= r {
		//     mid := l + (r - l) / 2
		//     if pre[mid] - pre[i] + nums[i] >= target {
		//         pos = mid
		//         r = mid - 1
		//     } else {
		//         l = mid + 1
		//     }
		// }
		// if pos != i - 1 {
		//     ans = min(ans, pos - i + 1)
		// }
		tar := target
		if i > 0 {
			tar += pre[i-1]
		}
		index := sort.SearchInts(pre, tar)
		if index >= i && index < n {
			ans = min(ans, index-i+1)
		}

	}
	return ans
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// 岛屿数量  深搜DFS
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
		for j := range visited[i] {
			visited[i][j] = false
		}
	}
	var dfs func(int, int)
	dfs = func(x, y int) {
		visited[x][y] = true
		next := [][]int{{-1, 0}, {1, 0}, {0, 1}, {0, -1}}
		for i := range next {
			nx, ny := x+next[i][0], y+next[i][1]
			if nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == byte('1') && !visited[nx][ny] {
				dfs(nx, ny)
			}
		}
	}
	ans := 0
	for i := range grid {
		for j := range grid[0] {
			if grid[i][j] == byte('1') && !visited[i][j] {
				ans++
				dfs(i, j)
			}
		}
	}
	return ans
}

// 股票价格跨度 单调栈
type StockSpanner struct {
	stack [][2]int
	idx   int
}

func Constructor() StockSpanner {
	return StockSpanner{[][2]int{{-1, math.MaxInt32}}, -1}
}

func (this *StockSpanner) Next(price int) int {
	this.idx++
	for price >= this.stack[len(this.stack)-1][1] {
		this.stack = this.stack[:len(this.stack)-1]
	}
	this.stack = append(this.stack, [2]int{this.idx, price})
	return this.idx - this.stack[len(this.stack)-2][0]
}

/**
 * Your StockSpanner object will be instantiated and called as such:
 * obj := Constructor();
 * param_1 := obj.Next(price);
 */

// 课程表 深度优先搜索dfs
func canFinish(numCourses int, prerequisites [][]int) bool {
	// res := []int{}
	dic := make([][]int, numCourses)
	for _, v := range prerequisites {
		dic[v[1]] = append(dic[v[1]], v[0])
	}
	stat := make([]int, numCourses)
	valid := true
	var dfs func(int)
	dfs = func(idx int) {
		stat[idx] = 1
		for _, v := range dic[idx] {
			if stat[v] == 1 {
				valid = false
				return
			} else if stat[v] == 0 {
				dfs(v)
				if !valid {
					return
				}
			}
		}
		// res = append(res, idx)
		stat[idx] = 2
	}
	for i, v := range stat {
		if v == 0 {
			dfs(i)
			if !valid {
				return false
				// return []int{}
			}
		}
	}
	return valid
	// for i := 0; i < numCourses/2; i++ {
	// 	res[i], res[numCourses-1-i] = res[numCourses-1-i], res[i]
	// }
	// return res
}

// 课程表 广度优先搜索 bfs
func canFinish2(numCourses int, prerequisites [][]int) bool {
	dic := make([][]int, numCourses)
	indeg := make([]int, numCourses)
	count := 0
	// res := []int{}
	for _, v := range prerequisites {
		dic[v[1]] = append(dic[v[1]], v[0])
		indeg[v[0]]++
	}
	q := []int{}
	for i, v := range indeg {
		if v == 0 {
			q = append(q, i)
		}
	}
	for len(q) > 0 {
		tNode := q[0]
		q = q[1:]
		count++
		// res = append(res, tNode)
		for _, v := range dic[tNode] {
			indeg[v]--
			if indeg[v] == 0 {
				q = append(q, v)
			}
		}
	}
	return count == numCourses
	// if len(res) < numCourses {
	//     return []int{}
	// }
	// return res
}

// 最长递增子序列
func lengthOfLIS(nums []int) int {
	n := len(nums)
	arr := []int{nums[0]}
	len := 1
	for i := 1; i < n; i++ {
		if nums[i] > arr[len-1] {
			arr = append(arr, nums[i])
			len++
		} else {
			// l, r, pos := 0, len - 1, -1
			// for l <= r {
			//     mid := l + (r - l) / 2
			//     if nums[i] > arr[mid] {
			//         pos = mid
			//         l = mid + 1
			//     } else {
			//         r = mid - 1
			//     }
			// }
			// arr[pos + 1] = nums[i]

			// k := sort.SearchInts(arr, nums[i])

			k := sort.Search(len, func(j int) bool { return nums[i] <= arr[j] }) // 同上
			arr[k] = nums[i]
		}
	}
	return len
}

// 最长递增子序列的个数
// https://leetcode.cn/problems/number-of-longest-increasing-subsequence/
func findNumberOfLIS(nums []int) int {
	d := [][]int{} // d[i]存储的是最长递增子序列长度为i+1的历史结尾数字，非递增(d[i][j] >= d[i][j + 1])
	cnt := [][]int{}
	for _, v := range nums {
		i := sort.Search(len(d), func(x int) bool { return d[x][len(d[x])-1] >= v })
		c := 1
		if i > 0 {
			k := sort.Search(len(d[i-1]), func(x int) bool { return d[i-1][x] < v })
			c = cnt[i-1][len(cnt[i-1])-1] - cnt[i-1][k]
		}
		if i == len(d) {
			d = append(d, []int{v})
			cnt = append(cnt, []int{0, c}) // 前缀和优化
		} else {
			d[i] = append(d[i], v)
			cnt[i] = append(cnt[i], cnt[i][len(cnt[i])-1]+c)
		}
	}
	return cnt[len(cnt)-1][len(cnt[len(cnt)-1])-1]
}

// 袋子里最少数目的球
// 题目为：给定拆分次数，寻找单个袋子球数的最大值
// 将题目转化为：给定单个袋子球数的最大值，计算拆分次数（如果一个袋子最多只能装x个球，需要拆分多少次？）
func minimumSize(nums []int, maxOperations int) int {
	r := 1
	for _, v := range nums {
		r = max(r, v)
	}
	dfs := func(i int) (ans int) {
		if i == 0 {
			return math.MaxInt
		}
		for _, v := range nums {
			ans += (v - 1) / i
		}
		return
	}

	// return sort.Search(r, func(i int) bool {return dfs(i) <= maxOperations})

	l, pos := 1, 0
	for l <= r {
		mid := l + (r-l)/2
		if dfs(mid) > maxOperations {
			pos = mid
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return pos + 1
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 两球之间的磁力   二分
func maxDistance(position []int, m int) int {
	n := len(position)
	sort.Ints(position)
	f := func(x int) int {
		if x == 0 {
			return math.MaxInt
		}
		t, r := 0, 0 // t为当前装了多少个球，下一个球放置位置必须大于r的值
		for _, v := range position {
			if v >= r {
				t++
				r = v + x
			}
		}
		return t
	}
	return sort.Search(position[n-1]-position[0]+1, func(x int) bool { return f(x) < m }) - 1
}

// 地图分析 https://leetcode.cn/problems/as-far-from-land-as-possible/
// 暴力bfs 会tle
func maxDistance2(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	next := [4][2]int{{-1, 0}, {1, 0}, {0, 1}, {0, -1}}
	bfs := func(x, y int) int {
		visited := make([][]bool, m)
		for i := range visited {
			visited[i] = make([]bool, n)
			for j := range visited[i] {
				visited[i][j] = false
			}
		}
		queue := [][2]int{{x, y}}
		visited[x][y] = true
		count := 0
		for len(queue) > 0 {
			count++
			tn := len(queue)
			for i := 0; i < tn; i++ {
				tx, ty := queue[0][0], queue[0][1]
				for j := 0; j < 4; j++ {
					nx, ny := tx+next[j][0], ty+next[j][1]
					if nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] {
						if grid[nx][ny] == 1 {
							return count
						}
						visited[nx][ny] = true
						queue = append(queue, [2]int{nx, ny})
					}
				}
				queue = queue[1:]
			}
		}
		return -1
	}
	ans := -1
	for i := range grid {
		for j := range grid {
			if grid[i][j] == 0 {
				ans = max(ans, bfs(i, j))
			}
		}
	}
	return ans
}

// 地图分析 https://leetcode.cn/problems/as-far-from-land-as-possible/
// 堆优化的Dijkstra  多源最短路   但是这里因为边权为1, 所以完全可以不用优先队列，直接加到切片末尾，因为dis值小的点一定在切片前面
type hpNode struct {
	v, x, y int
}
type hp []*hpNode

func (h hp) Len() int            { return len(h) }
func (h hp) Less(i, j int) bool  { return h[i].v < h[j].v }
func (h hp) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *hp) Push(i interface{}) { *h = append(*h, i.(*hpNode)) }
func (h *hp) Pop() interface{} {
	ls := len(*h) - 1
	ans := (*h)[ls]
	*h = (*h)[:ls]
	return ans
}

func maxDistance3(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	next := [4][2]int{{-1, 0}, {1, 0}, {0, 1}, {0, -1}}
	d := make([][]int, m)
	for i := range d {
		d[i] = make([]int, n)
		for j := range d[i] {
			d[i][j] = math.MaxInt
		}
	}
	priorityQueue := hp{}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 {
				d[i][j] = 0
				priorityQueue = append(priorityQueue, &hpNode{0, i, j})
			}
		}
	}
	heap.Init(&priorityQueue)
	for len(priorityQueue) > 0 {
		tNode := heap.Pop(&priorityQueue).(*hpNode)
		for i := 0; i < 4; i++ {
			nx, ny := tNode.x+next[i][0], tNode.y+next[i][1]
			if nx >= 0 && nx < m && ny >= 0 && ny < n {
				if tNode.v+1 < d[nx][ny] {
					d[nx][ny] = tNode.v + 1
					heap.Push(&priorityQueue, &hpNode{d[nx][ny], nx, ny})
				}
			}
		}
	}
	ans := -1
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 0 && d[i][j] != math.MaxInt {
				ans = max(ans, d[i][j])
			}
		}
	}
	return ans
}

// 地图分析 https://leetcode.cn/problems/as-far-from-land-as-possible/
// 多源bfs
func maxDistance4(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	next := [4][2]int{{-1, 0}, {1, 0}, {0, 1}, {0, -1}}
	d := make([][]int, m)
	for i := range d {
		d[i] = make([]int, n)
		for j := range d[i] {
			d[i][j] = math.MaxInt
		}
	}
	queue := [][2]int{}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 {
				d[i][j] = 0
				queue = append(queue, [2]int{i, j})
			}
		}
	}
	for len(queue) > 0 {
		tNode := queue[0]
		queue = queue[1:]
		for i := 0; i < 4; i++ {
			nx, ny := tNode[0]+next[i][0], tNode[1]+next[i][1]
			if nx >= 0 && nx < m && ny >= 0 && ny < n {
				if d[tNode[0]][tNode[1]]+1 < d[nx][ny] {
					d[nx][ny] = d[tNode[0]][tNode[1]] + 1
					queue = append(queue, [2]int{nx, ny})
				}
			}
		}
	}
	ans := -1
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 0 && d[i][j] != math.MaxInt {
				ans = max(ans, d[i][j])
			}
		}
	}
	return ans
}

// 判断子序列 动态规划，预处理出对于 t 的每一个位置，从该位置开始往后每一个字符第一次出现的位置。
// https://leetcode.cn/problems/is-subsequence/
func isSubsequence(s string, t string) bool {
	m, n := len(s), len(t)
	dp := make([][26]int, n+1)
	for i := 0; i < 26; i++ {
		dp[n][i] = n
	}
	for i := n - 1; i >= 0; i-- {
		for j := 0; j < 26; j++ {
			if int(t[i]) == int(j+'a') {
				dp[i][j] = i
			} else {
				dp[i][j] = dp[i+1][j]
			}
		}
	}
	trav := 0
	for i := 0; i < m; i++ {
		if dp[trav][int(s[i]-'a')] == n {
			return false
		}
		trav = dp[trav][int(s[i]-'a')] + 1
	}
	return true
}

// 补给覆盖  dp
// https://leetcode.cn/contest/hhrc2022/problems/wFtovi/
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

// a : 自己没有，父亲有，儿子不能有，孙子有
// b : 自己没有，儿子有
// c : 自己有
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func minSupplyStationNumber(root *TreeNode) int {
	var dfs func(*TreeNode) (int, int, int)
	dfs = func(root *TreeNode) (int, int, int) {
		if root == nil {
			return 0, 0, 10000
		}
		la, lb, lc := dfs(root.Left)
		ra, rb, rc := dfs(root.Right)
		a := lb + rb
		b := min(lc+rc, min(lc+rb, lb+rc))
		c := 1 + min(la, min(lb, lc)) + min(ra, min(rb, rc))
		return a, b, c
	}
	_, b, c := dfs(root)
	return min(b, c)
}

// priority_queue for int 优先队列堆int模板
// type hp struct{ sort.IntSlice }

// func (h hp) Less(i, j int) bool  { return h.IntSlice[i] > h.IntSlice[j] }
// func (h *hp) Push(i interface{}) { h.IntSlice = append(h.IntSlice, i.(int)) }
// func (h *hp) Pop() interface{} {
// 	ls := len(h.IntSlice) - 1
// 	ans := h.IntSlice[ls]
// 	h.IntSlice = h.IntSlice[:ls]
// 	return ans
// }
// func funcxxx(material []int) int {
// 	q := &hp{[]int{}}
// 	q.IntSlice = append(q.IntSlice, material...)
// 	heap.Init(q)

// 	a := heap.Pop(q).(int)
// 	heap.Push(q, a)
// }

// 二叉树中序遍历非递归
func inorderTraversal(root *TreeNode) (ans []int) {
	sta := []*TreeNode{}
	trav := root
	for len(sta) > 0 || trav != nil {
		for trav != nil {
			sta = append(sta, trav)
			trav = trav.Left
		}
		trav = sta[len(sta)-1]
		sta = sta[:len(sta)-1]
		ans = append(ans, trav.Val)
		trav = trav.Right
	}
	return
}

// 二叉树前序遍历非递归
func preorderTraversal(root *TreeNode) (ans []int) {
	sta := []*TreeNode{}
	trav := root
	for len(sta) > 0 || trav != nil {
		for trav != nil {
			ans = append(ans, trav.Val)
			sta = append(sta, trav)
			trav = trav.Left
		}
		trav = sta[len(sta)-1]
		sta = sta[:len(sta)-1]
		trav = trav.Right
	}
	return
}

// 二叉树的后序遍历非递归
func postorderTraversal(root *TreeNode) (ans []int) {
	dic := map[*TreeNode]int{}
	sta := []*TreeNode{}
	trav := root
	for len(sta) > 0 || trav != nil {
		for trav != nil {
			sta = append(sta, trav)
			trav = trav.Left
		}
		for len(sta) > 0 {
			trav = sta[len(sta)-1]
			if _, ok := dic[trav]; ok {
				sta = sta[:len(sta)-1]
				ans = append(ans, trav.Val)
			} else {
				dic[trav] = 1
				break
			}
		}
		if len(sta) == 0 {
			return
		}

		trav = trav.Right
	}
	return
}

// 字母异位词分组
// https://leetcode.cn/problems/group-anagrams/
func groupAnagrams(strs []string) (ans [][]string) {
	dic := map[[26]int][]string{} // map的key不能是slice, 但可以是数组
	f := func(s string) (fans [26]int) {
		for _, ch := range s {
			fans[int(ch-'a')]++
		}
		return
	}
	for _, str := range strs {
		tv := f(str)
		dic[tv] = append(dic[tv], str) // 第一次append时，dic[tv]为默认值[]string{}
	}
	for _, v := range dic {
		ans = append(ans, v)
	}
	return
}

// 将数组分成三个子数组的方案数   二分找右边界时不能提前return
// https://leetcode.cn/problems/ways-to-split-array-into-three-subarrays/
func waysToSplit(nums []int) (ans int) {
	MOD := int(1e9 + 7)
	n := len(nums)
	pre := make([]int, n+1)
	for i := range nums {
		pre[i+1] = pre[i] + nums[i]
	}
	for i := 1; i <= n-2; i++ {
		l := sort.Search(n, func(k int) bool {
			if k <= i {
				return false
			}
			return pre[k]-pre[i] >= pre[i]
		})
		if l == n {
			return
		}

		r := sort.Search(n, func(k int) bool {
			if k < l {
				return false
			}
			return pre[n]-pre[k] < pre[k]-pre[i]
		})
		// if r == l {
		//     return
		// }
		ans = (ans + r - l) % MOD
	}
	return
}

// 将数组分成三个子数组的方案数  i, l  i, r两对双指针，i增大时，l, r随着i增大而增大，特别注意nums全为0的情况，不处理的话l会小于i
// https://leetcode.cn/problems/ways-to-split-array-into-three-subarrays/
func waysToSplit2(nums []int) (ans int) {
	MOD := int(1e9 + 7)
	n := len(nums)
	pre := make([]int, n+1)
	for i := range nums {
		pre[i+1] = pre[i] + nums[i]
	}
	l, r := 2, 3
	for i := 1; i <= n-2; i++ {
		for l <= n-1 && pre[l]-pre[i] < pre[i] {
			l++
		}
		if l <= i { // 处理nums全为0的情况
			l = i + 1
		}
		if l == n {
			return
		}
		for r <= n-1 && pre[n]-pre[r] >= pre[r]-pre[i] {
			r++
		}
		if r >= l && pre[n]-pre[r-1] >= pre[l]-pre[i] {
			ans = (ans + r - l) % MOD
		} else {
			return
		}
	}
	return
}

// 颜色交替的最短路径  bfs 分别记录红边、蓝边到达当前节点的最短路径
// https://leetcode.cn/problems/shortest-path-with-alternating-colors/
func shortestAlternatingPaths(n int, redEdges [][]int, blueEdges [][]int) (res []int) {
	RED, BLUE := 0, 1
	ans := make([][2]int, n)
	for i := 1; i < n; i++ {
		ans[i][RED], ans[i][BLUE] = math.MaxInt, math.MaxInt
	}

	dic := make([][2][]int, n)
	for _, v := range redEdges {
		dic[v[0]][RED] = append(dic[v[0]][RED], v[1])
	}
	for _, v := range blueEdges {
		dic[v[0]][BLUE] = append(dic[v[0]][BLUE], v[1])
	}

	queue := [][2]int{{0, RED}, {0, BLUE}}

	for len(queue) > 0 {
		t := queue[0]
		queue = queue[1:]
		for _, v := range dic[t[0]][t[1]] {
			if ans[t[0]][1-t[1]]+1 < ans[v][t[1]] {
				ans[v][t[1]] = ans[t[0]][1-t[1]] + 1
				queue = append(queue, [2]int{v, 1 - t[1]})
			}
		}
	}
	for i := 0; i < n; i++ {
		t := min(ans[i][RED], ans[i][BLUE])
		if t == math.MaxInt {
			res = append(res, -1)
		} else {
			res = append(res, t)
		}
	}
	return
}

// 访问所有节点的最短路径   状态压缩 + bfs
// https://leetcode.cn/problems/shortest-path-visiting-all-nodes/
func shortestPathLength(graph [][]int) int {
	n := len(graph)
	memo := make([][]bool, n)
	for i := range memo {
		memo[i] = make([]bool, 1<<n)
	}
	type tuple struct{ u, mask, distance int }
	queue := []tuple{}
	for i := range graph {
		queue = append(queue, tuple{i, 1 << i, 0})
		memo[i][1<<i] = true
	}
	for {
		t := queue[0]
		queue = queue[1:]
		if t.mask == 1<<n-1 {
			return t.distance
		}
		for _, v := range graph[t.u] {
			tmask := t.mask | (1 << v)
			if !memo[v][tmask] {
				memo[v][tmask] = true
				queue = append(queue, tuple{v, tmask, t.distance + 1})
			}
		}
	}
}

// string []byte
// str := "adf"
// tbyte := []byte(str)
// tbyte[1] = 'e'
// tbyte = append(tbyte, 'q')
// tstr := string(tbyte[:])
// fmt.Println(tstr, int('a'), int('A'))
// https://leetcode.cn/problems/letter-case-permutation/ 字母大小写全排列
func letterCasePermutation(s string) (ans []string) {
	var dfs func([]byte, int)
	dfs = func(s []byte, idx int) {
		if idx == len(s) {
			ans = append(ans, string(s[:]))
			return
		}
		if unicode.IsDigit(rune(s[idx])) {
			dfs(s, idx+1)
		} else {
			dfs(s, idx+1)
			s[idx] ^= 32
			dfs(s, idx+1)
			// s[idx] ^= 32
		}
	}
	dfs([]byte(s), 0)
	return
}

// 最小基因变化 bfs
// https://leetcode.cn/problems/minimum-genetic-mutation/
func minMutation(start string, end string, bank []string) int {
	can := map[string]int{}
	for _, s := range bank {
		can[s] = 1
	}
	if can[end] != 1 {
		return -1
	}
	memo := map[string]int{}
	memo[start] = 1
	change := []byte{'A', 'C', 'G', 'T'}
	type node struct {
		s    string
		step int
	}
	queue := []node{{start, 0}}
	for len(queue) > 0 {
		t := queue[0]
		queue = queue[1:]
		if t.s == end {
			return t.step
		}
		tb := []byte(t.s)
		for i := 0; i < 8; i++ {
			old := tb[i]
			for _, v := range change {
				tb[i] = v
				ts := string(tb[:])
				if can[ts] == 1 && memo[ts] == 0 {
					memo[ts] = 1
					queue = append(queue, node{ts, t.step + 1})
				}
			}
			tb[i] = old
		}

	}
	return -1
}

// 按权重随机选择  前缀和 + 二分
// https://leetcode.cn/problems/random-pick-with-weight/
/**
 * Your Solution object will be instantiated and called as such:
 * obj := ConstructorPickIndex(w);
 * param_1 := obj.PickIndex();
 */
type Solution struct {
	pre []int
}

func ConstructorPickIndex(w []int) Solution {
	for i := 1; i < len(w); i++ {
		w[i] += w[i-1]
	}
	return Solution{w}
}

func (this *Solution) PickIndex() int {
	randx := rand.Intn(this.pre[len(this.pre)-1]) + 1
	return sort.SearchInts(this.pre, randx)
}

// 无重叠区间 贪心
// https://leetcode.cn/problems/non-overlapping-intervals/
func eraseOverlapIntervals(intervals [][]int) int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][1] < intervals[j][1]
	})
	ans, right := 1, intervals[0][1]
	for _, v := range intervals[1:] {
		if v[0] >= right {
			ans++
			right = v[1]
		}
	}
	return len(intervals) - ans
}

// 单词接龙 优化建图 + 双向bfs模板
// https://leetcode.cn/problems/word-ladder/
func ladderLength(beginWord string, endWord string, wordList []string) int {
	wordDic := map[string]int{}
	graph := [][]int{}
	addWord := func(s string) int {
		id, ok := wordDic[s]
		if !ok {
			id = len(wordDic)
			wordDic[s] = id
			graph = append(graph, []int{})
		}
		return id
	}

	addEdge := func(s string) int {
		id1 := addWord(s)
		tb := []byte(s)
		for i, b := range tb {
			tb[i] = '*'
			ts := string(tb[:])
			id2 := addWord(ts)
			graph[id1] = append(graph[id1], id2)
			graph[id2] = append(graph[id2], id1)
			tb[i] = b
		}
		return id1
	}

	for _, v := range wordList {
		addEdge(v)
	}

	endId, ok := wordDic[endWord]
	if !ok {
		return 0
	}
	beginId := addEdge(beginWord)

	const INF int = math.MaxInt64
	wordAmount := len(wordDic)
	// distToBegin、queueToBegin是为end节点维护的距离切片和队列，distToEnd、queueToEnd是为begin节点维护的距离切片和队列
	distToBegin, distToEnd := make([]int, wordAmount), make([]int, wordAmount)
	for i := range distToBegin {
		distToBegin[i] = INF
		distToEnd[i] = INF
	}
	distToBegin[endId] = 0
	distToEnd[beginId] = 0
	queueToBegin, queueToEnd := []int{endId}, []int{beginId}

	// for len(queueToBegin) > 0 && len(queueToEnd) > 0 {
	//     t := queueToEnd[0]
	//     queueToEnd = queueToEnd[1:]
	//     if distToBegin[t] < INF {
	//         return (distToEnd[t] + distToBegin[t]) / 2 + 1
	//     }
	//     for _, v := range graph[t] {
	//         if distToEnd[t] + 1 < distToEnd[v] {
	//             distToEnd[v] = distToEnd[t] + 1
	//             queueToEnd = append(queueToEnd, v)
	//         }
	//     }

	//     t = queueToBegin[0]
	//     queueToBegin = queueToBegin[1:]
	//     if distToEnd[t] < INF {
	//         return (distToEnd[t] + distToBegin[t]) / 2 + 1
	//     }
	//     for _, v := range graph[t] {
	//         if distToBegin[t] + 1 < distToBegin[v] {
	//             distToBegin[v] = distToBegin[t] + 1
	//             queueToBegin = append(queueToBegin, v)
	//         }
	//     }
	// }

	// 上面是双向一个一个遍历，下面是双向一层一层遍历
	for len(queueToBegin) > 0 && len(queueToEnd) > 0 {
		tq := queueToEnd
		queueToEnd = nil
		for _, t := range tq {
			if distToBegin[t] < INF {
				return (distToEnd[t]+distToBegin[t])/2 + 1
			}
			for _, v := range graph[t] {
				if distToEnd[v] == INF {
					distToEnd[v] = distToEnd[t] + 1
					queueToEnd = append(queueToEnd, v)
				}
			}
		}

		tq = queueToBegin
		queueToBegin = nil
		for _, t := range tq {
			if distToEnd[t] < INF {
				return (distToEnd[t]+distToBegin[t])/2 + 1
			}
			for _, v := range graph[t] {
				if distToBegin[v] == INF {
					distToBegin[v] = distToBegin[t] + 1
					queueToBegin = append(queueToBegin, v)
				}
			}
		}
	}

	return 0
}

// 删除最短的子数组使剩余数组有序 预处理找到i, j，以使[0, i) [j, n)有序，一种做法是固定i，二分j，另一种做法是滑动窗口
// https://leetcode.cn/problems/shortest-subarray-to-be-removed-to-make-array-sorted/
func findLengthOfShortestSubarray(arr []int) int {
	n := len(arr)
	i, j := 1, n-1
	for i < n && arr[i-1] <= arr[i] {
		i++
	}
	if i == n {
		return 0
	}
	for j-1 >= 0 && arr[j-1] <= arr[j] {
		j--
	}
	ans := j
	// for k := 0; k < i; k++ {
	// 	idx := sort.SearchInts(arr[j:], arr[k]) + j
	// 	ans = min(ans, idx-k-1)
	// }
	l, r := 0, j
	for l < i && r < n {
		if arr[l] <= arr[r] {
			ans = min(ans, r-l-1)
			l++
		} else {
			r++
		}
	}
	ans = min(ans, n-i)
	return ans
}

// 元素和小于等于阈值的正方形的最大边长  二维前缀和 + 二分正方形边长
// https://leetcode.cn/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/
func maxSideLength(mat [][]int, threshold int) int {
	m, n := len(mat), len(mat[0])
	sum := make([][]int, m+1)
	for i := 0; i < m+1; i++ {
		sum[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			sum[i][j] = sum[i-1][j] + sum[i][j-1] - sum[i-1][j-1] + mat[i-1][j-1]
		}
	}
	f := func(x int) bool {
		if x == 0 {
			return true
		}
		x--
		for i := 1; i <= m-x; i++ {
			for j := 1; j <= n-x; j++ {
				tsum := sum[i+x][j+x] - sum[i+x][j-1] - sum[i-1][j+x] + sum[i-1][j-1]
				if tsum <= threshold {
					return true
				}
			}
		}
		return false
	}
	r := min(m, n)
	return sort.Search(r+1, func(k int) bool { return f(k) == false }) - 1
}

// 满足条件的子序列数目 一个序列的最大最小值与序列顺序无关，先预处理2的幂，然后排序，最后二分或者双指针
// https://leetcode.cn/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/
// 给你一个整数数组 nums 和一个整数 target 。
// 请你统计并返回 nums 中能满足其最小元素与最大元素的 和 小于或等于 target 的 非空 子序列的数目。
// 由于答案可能很大，请将结果对 109 + 7 取余后返回。
func numSubseq(nums []int, target int) (ans int) {
	const MOD int = 1e9 + 7
	MaxN := len(nums)
	f := make([]int, MaxN)
	f[0] = 1
	for i := 1; i < MaxN; i++ {
		f[i] = (f[i-1] * 2) % MOD
	}
	sort.Ints(nums)
	// for i := 0; i < MaxN && nums[i] * 2 <= target; i++ {
	//     upperBound := target - nums[i]
	//     idx := sort.Search(MaxN, func(x int) bool {return nums[x] > upperBound}) - 1 - i
	//     ans = (ans + f[idx]) % MOD
	// }
	l, r := 0, MaxN-1
	for l <= r {
		if nums[l]+nums[r] > target {
			r--
		} else {
			ans = (ans + f[r-l]) % MOD
			l++
		}
	}
	return
}

// 删除并获得点数  原地去重 + 记录相同值的右侧端点 + 打家劫舍类似的dp + 判断是否是连续的，若不连续可以取相邻的值
// https://leetcode.cn/problems/delete-and-earn/
func deleteAndEarn(nums []int) int {
	sort.Ints(nums)
	dic := map[int]int{} // 记录相同值的右侧端点
	k, n := 0, len(nums)
	for i := 1; i < n; i++ {
		if nums[i] != nums[k] {
			k++
			nums[k] = nums[i]
		}
		dic[nums[k]] = i

	}
	k++
	// fmt.Println(nums, k, dic)
	if k == 1 {
		return nums[0] * (dic[nums[0]] + 1)
	}
	a, b := 0, 0
	if nums[0]+1 == nums[1] {
		a, b = nums[0]*(dic[nums[0]]+1), max(nums[0]*(dic[nums[0]]+1), nums[1]*(dic[nums[1]]-dic[nums[0]]))
	} else {
		a, b = nums[0]*(dic[nums[0]]+1), nums[0]*(dic[nums[0]]+1)+nums[1]*(dic[nums[1]]-dic[nums[0]])
	}

	for i, v := range nums[2:k] {
		if v == nums[i+1]+1 {
			a, b = b, max(b, a+v*(dic[nums[i+2]]-dic[nums[i+1]]))
		} else {
			a, b = b, b+v*(dic[nums[i+2]]-dic[nums[i+1]])
		}

	}
	return b
}

// 按位或最大的最小子数组长度    利用或运算的性质 + 通用模板
// https://leetcode.cn/problems/smallest-subarrays-with-maximum-bitwise-or/
// 可求出所有子数组的按位或的结果，以及值等于该结果的子数组的个数。
// 可求按位或结果等于任意给定数字的子数组的最短长度/最长长度。
func smallestSubarrays(nums []int) []int {
	type ornode struct{ or, i int }
	ors := []ornode{} // 按位或的值 + 对应子数组的右端点的最小值
	n := len(nums)
	ans := make([]int, n)
	for i := n - 1; i >= 0; i-- {
		ors = append(ors, ornode{0, i})
		ors[0].or |= nums[i]
		k := 0
		for _, p := range ors[1:] {
			p.or |= nums[i]
			if p.or == ors[k].or {
				ors[k].i = p.i // 合并相同值，下标取最小的
			} else {
				k++
				ors[k] = p
			}
		}
		ors = ors[:k+1]
		// fmt.Println(ors)
		// 本题只用到了 ors[0]，如果题目改成任意给定数字，可以在 ors 中查找
		ans[i] = ors[0].i - i + 1
	}
	return ans
}

// 找到最接近目标值的函数值    直接用上题的模板
// https://leetcode.cn/problems/find-a-value-of-a-mysterious-function-closest-to-target/
func closestToTarget(arr []int, target int) int {
	n := len(arr)
	type pair struct{ or, i int }
	ors := []pair{}
	ans := math.MaxInt
	for i := n - 1; i >= 0; i-- {
		ors = append(ors, pair{arr[i], i}) // 注意这里不能是{0，i}，不然 0 & 任何数 都是0
		ors[0].or &= arr[i]
		k := 0
		for _, p := range ors[1:] {
			p.or &= arr[i]
			if p.or == ors[k].or {
				ors[k].i = p.i
			} else {
				k++
				ors[k] = p
			}
		}
		ors = ors[:k+1]
		idx := sort.Search(k+1, func(x int) bool { return ors[x].or > target })
		if idx < k+1 {
			ans = min(ans, ors[idx].or-target)
		}
		if idx-1 >= 0 {
			ans = min(ans, target-ors[idx-1].or)
		}
		// fmt.Println(ors, ans)
	}
	return ans
}

// 快照数组 快照id增加数据可以暂时不变，等到set时判断版本号是否一致，如果不一致再在特定位置追加新的值，lazy思想
// https://leetcode.cn/problems/snapshot-array/
/**
 * Your SnapshotArray object will be instantiated and called as such:
 * obj := SnapshotArrayConstructor(length);
 * obj.Set(index,val);
 * param_2 := obj.Snap();
 * param_3 := obj.Get(index,snap_id);
 */
type SnapshotArray struct {
	version int
	ids     [][]int
	vals    [][]int
}

func SnapshotArrayConstructor(length int) SnapshotArray {
	ids, vals := make([][]int, length), make([][]int, length)
	for i := 0; i < length; i++ {
		ids[i] = []int{0}
		vals[i] = []int{0}
	}
	return SnapshotArray{0, ids, vals}
}

func (this *SnapshotArray) Set(index int, val int) {
	n := len(this.ids[index])
	if this.version > this.ids[index][n-1] {
		this.ids[index] = append(this.ids[index], this.version)
		this.vals[index] = append(this.vals[index], val)
	} else {
		this.vals[index][n-1] = val
	}
}

func (this *SnapshotArray) Snap() int {
	this.version++
	return this.version - 1
}

func (this *SnapshotArray) Get(index int, snap_id int) int {
	return this.vals[index][sort.SearchInts(this.ids[index], snap_id+1)-1]
}

// 环形子数组的最大和 分两种情况 注意判断全为负的情况
// https://leetcode.cn/problems/maximum-sum-circular-subarray/
func maxSubarraySumCircular(nums []int) int {
	total, currMax, currMin, maxSum, minSum := nums[0], nums[0], nums[0], nums[0], nums[0]
	for _, v := range nums[1:] {
		total += v
		currMax = max(v, currMax+v)
		maxSum = max(maxSum, currMax)
		currMin = min(v, currMin+v)
		minSum = min(minSum, currMin)
	}
	if maxSum < 0 {
		return maxSum
	}
	return max(maxSum, total-minSum)
}

// 丑数II 丑数为2^x * 3^y * 5^z 的数，枚举x,y,z 二分找x,小于x的数中有n个丑数
// https://leetcode.cn/problems/ugly-number-ii/
func nthUglyNumber(n int) int {
	f := func(x int) int {
		ans := 0
		for ix := 1; ix < x; ix *= 2 {
			for iy := 1; iy < x && ix*iy < x; iy *= 3 {
				for iz := 1; iz < x && ix*iy*iz < x; iz *= 5 {
					ans++
				}
			}
		}
		return ans
	}
	return sort.Search(1e10, func(x int) bool { return f(x) >= n }) - 1
}

// 丑数III 最小公倍数
// https://leetcode.cn/problems/ugly-number-iii/
func nthUglyNumberIII(n int, a int, b int, c int) int {
	var gcd func(int, int) int
	gcd = func(x, y int) int {
		if x%y == 0 {
			return y
		}
		return gcd(y, x%y)
	}
	f := func(x int) int {
		ab := (a * b) / gcd(a, b)
		ac := (a * c) / gcd(a, c)
		bc := (b * c) / gcd(b, c)
		abc := (ab * c) / gcd(ab, c)
		return (x-1)/a + (x-1)/b + (x-1)/c - (x-1)/ab - (x-1)/ac - (x-1)/bc + (x-1)/abc
	}
	return sort.Search(2e9+1, func(x int) bool { return f(x) >= n }) - 1
}

// 多米诺和托米诺平铺   动规或者矩阵快速幂
// https://leetcode.cn/problems/domino-and-tromino-tiling/
func numTilings(n int) int {
	const MOD int = 1e9 + 7
	m := 4
	type matrix [4][4]int
	mul := func(a, b matrix) (c matrix) {
		for i := 0; i < m; i++ {
			for j := 0; j < m; j++ {
				for k := 0; k < m; k++ {
					c[i][j] = (c[i][j] + a[i][k]*b[k][j]%MOD) % MOD
				}
			}
		}
		return
	}

	quick_mul := func(a matrix, b int) matrix {
		base, ans := a, [4][4]int{}
		for i := 0; i < m; i++ {
			ans[i][i] = 1
		}
		for b > 0 {
			if b&1 == 1 {
				ans = mul(ans, base)
			}
			b >>= 1
			base = mul(base, base)
		}
		return ans
	}

	a := [4][4]int{
		{0, 0, 0, 1},
		{1, 0, 1, 0},
		{1, 1, 0, 0},
		{1, 1, 1, 1},
	}
	ans := quick_mul(a, n)
	return ans[3][3] % MOD

	// dp := make([][4]int, n + 1)
	// dp[0][3] = 1
	// for i := 1; i <= n; i++ {
	//     dp[i][0] = dp[i - 1][3]
	//     dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % MOD
	//     dp[i][2] = (dp[i - 1][0] + dp[i - 1][1]) % MOD
	//     dp[i][3] = (dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][3]) % MOD
	// }
	// return dp[n][3]

	// a, b, c, d := 0, 0, 0, 1
	// ta, tb, tc, td := 0, 0, 0, 0
	// for i := 1; i <= n; i++ {
	//     ta = d
	//     tb = (a + c) % MOD
	//     tc = (a + b) % MOD
	//     td = (a + b + c + d) % MOD
	//     a, b, c, d = ta, tb, tc, td
	// }
	// return d
}

// 前K个高频元素  快速选择
// https://leetcode.cn/problems/top-k-frequent-elements/
func topKFrequent(nums []int, k int) (ans []int) {
	dic := map[int]int{}
	data := []int{}
	for _, v := range nums {
		if dic[v] == 0 {
			data = append(data, v)
		}
		dic[v]++
	}
	n := len(data)
	f := func(l, r int) int {
		if l == r {
			return l
		}
		index := rand.Intn(r-l) + l // r - l > 0 否则会panic
		data[l], data[index] = data[index], data[l]
		pos := l
		for i := l + 1; i <= r; i++ {
			if dic[data[i]] > dic[data[l]] {
				pos++
				data[pos], data[i] = data[i], data[pos]
			}
		}
		data[pos], data[l] = data[l], data[pos]
		return pos
	}
	var sel func(int, int, int)
	sel = func(l, r, k int) {
		if l > r || k == 0 {
			return
		}
		pos := f(l, r)
		if pos-l+1 <= k {
			for i := l; i <= pos; i++ {
				ans = append(ans, data[i])
			}
			sel(pos+1, r, k-(pos-l+1))
		} else {
			sel(l, pos-1, k)
		}
	}
	sel(0, n-1, k)
	return
}

// 组合总和II
// https://leetcode.cn/problems/combination-sum-ii/
// 给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
// candidates 中的每个数字在每个组合中只能使用 一次 。
// 注意：解集不能包含重复的组合。

func combinationSum2(candidates []int, target int) (ans [][]int) {
	n := len(candidates)
	dic := map[int]int{}
	sort.Ints(candidates)
	for _, v := range candidates { // 记录每个数的出现次数
		dic[v]++
	}
	var dfs func(int, int, []int)
	dfs = func(x, cur int, ls []int) {
		if cur >= target || x == n {
			if cur == target {
				ans = append(ans, ls)
			}
			return
		}
		tn := dic[candidates[x]]
		for i := 0; i <= tn; i++ { // 递归时直接枚举当前数的选择个数，不论选择多少个都跳到下一个不重复数字(x + tn)继续递归
			dfs(x+tn, cur+candidates[x]*i, append([]int{}, ls...)) // 直接填ls会出错
			ls = append(ls, candidates[x])
		}
	}
	dfs(0, 0, []int{})
	return
}

// 组合总和II   法二    去重逻辑改为增加一个bool类型变量指示上一个数字是否被选取，另外ls辅助数组可以拿出来放到递归函数外面
// 在递归时，若发现没有选择上一个数，且当前数字与上一个数相同，则可以跳过当前生成的子集。
func combinationSum2_2(candidates []int, target int) (ans [][]int) {
	n := len(candidates)
	sort.Ints(candidates)
	ls := []int{}
	var dfs func(int, int, bool)
	dfs = func(x, cur int, pre bool) {
		if cur >= target || x == n {
			if cur == target {
				ans = append(ans, append([]int{}, ls...))
			}
			return
		}
		dfs(x+1, cur, false) // 不选当前数字
		if !pre && x > 0 && candidates[x] == candidates[x-1] {
			return
		}
		ls = append(ls, candidates[x]) // 选当前数字
		dfs(x+1, cur+candidates[x], true)
		ls = ls[:len(ls)-1]
	}
	dfs(0, 0, false)
	return
}

// 匹配子序列的单词数
// https://leetcode.cn/problems/number-of-matching-subsequences/
func numMatchingSubseq(s string, words []string) (ans int) {
	type pair struct{ i, j int } // i 为单词在words中的索引号，j为该单词已经匹配了的长度
	a := [26][]pair{}
	for i, w := range words {
		a[w[0]-'a'] = append(a[w[0]-'a'], pair{i, 0})
	}
	for _, v := range s { // 一次遍历s, 同时匹配words中的所有字符串
		p := a[v-'a']
		a[v-'a'] = nil
		for _, pa := range p {
			pa.j++
			if pa.j == len(words[pa.i]) {
				ans++
			} else {
				t := words[pa.i][pa.j] - 'a'
				a[t] = append(a[t], pa)
			}
		}
	}
	return
}

// sort.SearchInts()
// 求最长严格递增子序列需要二分找到大于或等于当前元素的元素位置（即 C++ 中的 lower_bound, go 为 sort.SearchInts(ls, v)
// 求最长不降子序列需要二分找到大于当前元素的元素位置（即 C++ 中的 upper_bound, go 为 sort.SearchInts(ls, v + 1)

// 若x是回文数字返回true
func palindromeCheck(x int) bool {
	t := []int{}
	for x > 0 {
		t = append(t, x%10)
		x /= 10
	}
	l, r := 0, len(t)-1
	for ; l < r; l, r = l+1, r-1 {
		if t[l] != t[r] {
			return false
		}
	}
	return true
}

// 回文素数
// https://leetcode.cn/problems/prime-palindrome/
func primePalindrome(n int) int {
	primeCheck := func(x int) bool {
		if x < 2 {
			return false
		}
		if x == 2 || x == 3 {
			return true
		}
		for i := 2; i*i <= x; i++ {
			if x%i == 0 {
				return false
			}
		}
		return true
	}
	powten := func(x int) int {
		ans := 1
		for i := 0; i < x; i++ {
			ans *= 10
		}
		return ans
	}
	for l := 0; l < 5; l++ { // 枚举所有的回文数字
		r := powten(l + 1)
		for root := powten(l); root < r; root++ { // 奇数回文数字, 长度为 l * 2 - 1
			s := strconv.Itoa(root)
			tb := []byte(s)
			for i := len(tb) - 2; i >= 0; i-- {
				tb = append(tb, tb[i])
			}
			tn, _ := strconv.Atoi(string(tb))
			if tn >= n && primeCheck(tn) {
				return tn
			}
		}
		for root := powten(l); root < r; root++ { // 偶数回文数字, 长度为 l * 2
			s := strconv.Itoa(root)
			tb := []byte(s)
			for i := len(tb) - 1; i >= 0; i-- {
				tb = append(tb, tb[i])
			}
			tn, _ := strconv.Atoi(string(tb))
			if tn >= n && primeCheck(tn) {
				return tn
			}
		}

	}
	return 0
}

// grid是一个二进制方形矩阵，1表示陆地，求岛屿面积的同时，每个岛屿都有特定的tag，这样对于任意一个节点，就能O(1)知道其所属岛屿的面积
func largestIsland(grid [][]int) (ans int) {
	n := len(grid)
	next := [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	dic := map[int]int{}
	var dfs func(int, int, int) int
	dfs = func(x, y, tag int) int {
		grid[x][y] = tag
		tot := 1
		for _, v := range next {
			nx, ny := x+v[0], y+v[1]
			if nx >= 0 && nx < n && ny >= 0 && ny < n && grid[nx][ny] == 1 {
				tot += dfs(nx, ny, tag)
			}
		}
		dic[tag] = tot // 回溯到最后时，dic[tag]记录的是整个岛屿的面积
		return tot
	}
	for i, row := range grid {
		for j, v := range row {
			if v == 1 {
				ans = max(dfs(i, j, 2+i*n+j), ans) // +2是为了避免tag为1
			}
		}
	}
	// do things
	return
}

// 不重复的全排列
// https://leetcode.cn/problems/permutations-ii/
func permuteUnique(nums []int) (ans [][]int) {
	n := len(nums)
	vis := make([]bool, n)
	sort.Ints(nums)
	var dfs func([]int)
	dfs = func(x []int) {
		if len(x) == n {
			ans = append(ans, append([]int{}, x...))
			return
		}
		last := -1
		for i, v := range nums {
			if !vis[i] && (last == -1 || v != nums[last]) {
				last = i
				vis[i] = true
				dfs(append(x, v))
				vis[i] = false
			}
		}
	}
	dfs([]int{})
	return
}

// 设计一个文本编辑器  双向链表
// https://leetcode.cn/problems/design-a-text-editor/
/**
 * Your TextEditor object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AddText(text);
 * param_2 := obj.DeleteText(k);
 * param_3 := obj.CursorLeft(k);
 * param_4 := obj.CursorRight(k);
 */
type TextEditor struct {
	*list.List
	cur *list.Element
}

func TextEditorConstructor() TextEditor {
	l := list.New()
	return TextEditor{l, l.PushBack(nil)}
}

// 将 text 添加到光标所在位置。添加完后光标在 text 的右边。
func (this *TextEditor) AddText(text string) {
	for _, ch := range text {
		this.cur = this.InsertAfter(byte(ch), this.cur)
	}
}

// 删除光标左边 k 个字符。返回实际删除的字符数目
func (this *TextEditor) DeleteText(k int) int {
	k0 := k
	for k > 0 && this.cur.Value != nil {
		pre := this.cur.Prev()
		this.Remove(this.cur)
		this.cur = pre
		k--
	}
	return k0 - k
}

// 返回光标左边 min(10, len) 个字符，其中 len 是光标左边的字符数目
func (this *TextEditor) Text() string {
	b := []byte{}
	for k, cur := 10, this.cur; k > 0 && cur.Value != nil; k, cur = k-1, cur.Prev() {
		b = append(b, cur.Value.(byte))
	}
	for i, n := 0, len(b); i < n/2; i++ {
		b[i], b[n-1-i] = b[n-1-i], b[i]
	}
	return string(b)
}

// 将光标向左移动 k 次。返回移动后光标左边 min(10, len) 个字符，其中 len 是光标左边的字符数目
func (this *TextEditor) CursorLeft(k int) string {
	for k > 0 && this.cur.Value != nil {
		this.cur = this.cur.Prev()
		k--
	}
	return this.Text()
}

// 将光标向右移动 k 次。返回移动后光标左边 min(10, len) 个字符，其中 len 是光标左边的字符数目
func (this *TextEditor) CursorRight(k int) string {
	for k > 0 && this.cur.Next() != nil {
		this.cur = this.cur.Next()
		k--
	}
	return this.Text()
}

// 下面是 设计文本编辑器 对顶栈 的做法
// type TextEditor struct {
// 	l, r []byte
// }

// func Constructor() TextEditor {
// 	return TextEditor{}
// }

// func (this *TextEditor) AddText(text string) {
// 	this.l = append(this.l, text...)
// }

// func (this *TextEditor) DeleteText(k int) int {
// 	n := len(this.l)
// 	if n > k {
// 		this.l = this.l[:n-k]
// 	} else {
// 		k = n
// 		this.l = this.l[:0]
// 	}
// 	return k
// }

// func (this *TextEditor) Text() string {
// 	n := len(this.l)
// 	if n > 10 {
// 		return string(this.l[n-10:])
// 	} else {
// 		return string(this.l[0:])
// 	}
// }

// func (this *TextEditor) CursorLeft(k int) string {
// 	for len(this.l) > 0 && k > 0 {
// 		this.r = append(this.r, this.l[len(this.l)-1])
// 		this.l = this.l[:len(this.l)-1]
// 		k--
// 	}
// 	return this.Text()
// }

// func (this *TextEditor) CursorRight(k int) string {
// 	for len(this.r) > 0 && k > 0 {
// 		this.l = append(this.l, this.r[len(this.r)-1])
// 		this.r = this.r[:len(this.r)-1]
// 		k--
// 	}
// 	return this.Text()
// }

// 0-1 背包问题的状态转换方程是：
// dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - v[i]] + w[i])
// 由于计算 dp[i][j] 的时候，依赖于 dp[i - 1][j - v[i]] 。
// 因此我们在改为「一维空间优化」时，需要确保 dp[j - v[i]] 存储的是上一行的值，即确保 dp[j - v[i]] 还没有被更新，
// 所以遍历方向是从大到小。
func maxValue1(n, c int, v, w []int) int {
	dp := make([]int, c+1)
	for i := 0; i < n; i++ {
		for j := c; j >= v[i]; j-- {
			dp[j] = max(dp[j], dp[j-v[i]]+w[i])
		}
	}
	return dp[c]
}

// 完全背包问题的状态转移方程是：
// dp[i][j] = max(dp[i - 1][j], dp[i][j - v[i]] + w[i])
// 由于计算 dp[i][j] 的时候，依赖于 dp[i][j - v[i]] 。
// 因此我们在改为「一维空间优化」时，需要确保 dp[j - v[i]] 存储的是当前行的值，即确保 dp[j - v[i]] 已经被更新，
// 所以遍历方向是从小到大。
// 形式上，我们只需要将 01 背包问题的「一维空间优化」解法中的「容量维度」遍历方向从「从大到小 改为 从小到大」就可以解决完全背包问题。
// 但本质是因为两者进行状态转移时依赖了不同的格子：
// 01 背包依赖的是「上一行正上方的格子」和「上一行左边的格子」。
// 完全背包依赖的是「上一行正上方的格子」和「本行左边的格子」。
func maxValue2(n, c int, v, w []int) int {
	dp := make([]int, c+1)
	for i := 0; i < n; i++ {
		for j := v[i]; j <= c; j++ {
			dp[j] = max(dp[j], dp[j-v[i]]+w[i])
		}
	}
	return dp[c]
}

// 将「多重背包」的多件物品进行「扁平化展开」，就转换成了「01 背包」
// 无论是「朴素二维」、「滚动数组」、「一维优化」还是「扁平化」都不能优化「多重背包」问题的时间复杂度。
// 在各维度数量级同阶的情况下，时间复杂度是 O(n^3) 的。
// 这意味着我们只能解决 10^2 数量级的「多重背包」问题。
// 同时，我们能总结出：在传统的三种背包问题的「一维空间优化」里，只有「完全背包」的「容量维度」是「从小到大」的，
// 其他两种背包的「容量维度」都是「从大到小」的。
// n 个物品，背包总容量为c, s[i]为第i个物品的最大选择数量，v[i]为第i个物品的体积，w[i]为第i个物品的价值
func maxValue3(n, c int, s, v, w []int) int {
	dp := make([]int, c+1)
	for i := 0; i < n; i++ {
		for j := c; j >= v[i]; j-- {
			for k := 1; k <= s[i] && k*v[i] <= j; k++ {
				dp[j] = max(dp[j], dp[j-k*v[i]]+k*w[i])
			}
		}
	}
	return dp[c]
}

// 二进制优化——将原本为 n 的物品用 ceil(log(n)) 个数来代替，从而降低算法复杂度。
// 二进制优化的本质，是对「物品」做分类，使得总数量为 n 的物品能够用更小的 logn 个数所组合表示出来。
// 经过「二进制优化」的多重背包单秒能解决的数据范围从 10^2 上升到 10^3 。
func maxValue3_1(n, c int, s, v, w []int) int {
	// 扁平化
	vv, ww := []int{}, []int{}
	// 我们希望每件物品都进行扁平化，所以首先遍历所有的物品
	for i := 0; i < n; i++ {
		// 获取每件物品的出现次数
		amount := s[i]
		// 进行扁平化：如果一件物品规定的使用次数为 7 次，我们将其扁平化为三件物品：1*重量&1*价值、2*重量&2*价值、4*重量&4*价值
		// 三件物品都不选对应了我们使用该物品 0 次的情况、只选择第一件扁平物品对应使用该物品 1 次的情况、只选择第二件扁平物品对应
		// 使用该物品 2 次的情况，只选择第一件和第二件扁平物品对应了使用该物品 3 次的情况 ...
		for k := 1; k <= amount; k *= 2 {
			amount -= k
			vv = append(vv, k*v[i])
			ww = append(ww, k*w[i])
		}
		if amount > 0 {
			vv = append(vv, amount*v[i])
			ww = append(ww, amount*w[i])
		}
	}
	// 0-1 背包问题解决方案
	dp := make([]int, c+1)
	for i := 0; i < n; i++ {
		for j := c; j >= v[i]; j-- {
			dp[j] = max(dp[j], dp[j-v[i]]+w[i])
		}
	}
	return dp[c]
}

// 多重背包单调队列优化
// 能够在转移 f[i] 时，以 O(1) 或者均摊 O(1) 的复杂度从「能够参与转移的状态」中找到最大值，我们就能省掉「朴素多重背包」解决方案中
// 最内层的“决策”循环，从而将整体复杂度降低到 O(N*C) 。
// 与对「物品」做拆分的「二进制优化」不同，「单调队列优化」是对「状态」做拆分操作。
// 利用某个状态必然是由余数相同的特定状态值转移而来进行优化。
func maxValue3_2(n, c int, s, v, w []int) int {
	// g为辅助队列，记录的是上一次的结果， q为主队列，记录的是本次的结果
	dp, g, q := make([]int, c+1), make([]int, c+1), make([]int, c+1)
	// 枚举物品
	for i := 0; i < n; i++ {
		si, vi, wi := s[i], v[i], w[i]
		// 将上次算的结果存入辅助数组中
		copy(g, dp)
		// 枚举余数
		for j := 0; j < vi; j++ {
			// 初始化队列，head 和 tail 分别指向队列头部和尾部
			head, tail := 0, -1
			// 枚举同一余数情况下，有多少种方案。
			// 例如余数为 1 的情况下有：1、vi + 1、2 * vi + 1、3 * vi + 1 ...
			for k := j; k <= c; k += vi {
				dp[k] = g[k]
				// 将不在窗口范围内的值弹出
				if head <= tail && k-q[head] > si*vi {
					head++
				}
				// 如果队列中存在元素，直接使用队头来更新
				if head <= tail {
					dp[k] = max(dp[k], g[q[head]]+(k-q[head])/vi*wi)
				}
				// 当前值比对尾值更优，队尾元素没有存在必要，队尾出队
				for head <= tail && g[q[tail]]+(j-q[tail])/vi*wi <= g[k]+(j-k)/vi*wi {
					tail--
				}
				// 将新下标入队
				tail++
				q[tail] = k
			}
		}
	}
	return dp[c]
}

// 混合背包
// 给定物品数量 n 和背包容量 c。第 i 件物品的体积是 v[i]，价值是 w[i]，可用数量为 s[i] ：
// 当 s[i] 为 -1 代表是该物品只能用一次
// 当 s[i] 为 0 代表该物品可以使用无限次
// 当 s[i] 为任意正整数则代表可用 s[i] 次
// 求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大。
func maxValue4(n, c int, s, v, w []int) int {
	// 构造出物品的「价值」和「体积」列表
	vv, ww := []int{}, []int{}
	for i := 0; i < n; i++ {
		amount := s[i]
		// 多重背包：应用「二进制优化」转换为 0-1 背包问题
		if amount > 0 {
			for k := 1; k <= amount; k *= 2 {
				amount -= k
				vv = append(vv, k*v[i])
				ww = append(ww, k*w[i])
			}
			if amount > 0 {
				vv = append(vv, amount*v[i])
				ww = append(ww, amount*w[i])
			}
		} else if amount == -1 { // 01 背包：直接添加
			vv = append(vv, v[i])
			ww = append(ww, w[i])
		} else { // 完全背包：对 worth 做翻转进行标记
			vv = append(vv, v[i])
			ww = append(ww, -w[i])
		}
	}
	// 使用「一维空间优化」方式求解三种背包问题
	dp := make([]int, c+1)
	for i := 0; i < n; i++ {
		// 01 背包：包括「原本的 01 背包」和「经过二进制优化的完全背包」
		// 容量「从大到小」进行遍历
		if w[i] >= 0 {
			for j := c; j >= v[i]; j-- {
				dp[j] = max(dp[j], dp[j-v[i]]+w[i])
			}
		} else { // 完全背包：容量「从小到大」进行遍历
			for j := v[i]; j <= c; j++ {
				// 同时记得将 worth 重新翻转为正整数
				dp[j] = max(dp[j], dp[j-v[i]]-w[i])
			}
		}

	}
	return dp[c]
}

// 分组背包
// 定义 f[i][j] 为考虑前 i 个物品组，背包容量不超过 j 的最大价值。
// 给定 n 个物品组，和容量为 c 的背包。
// 第 i 个物品组共有 s[i] 件物品，其中第 i 组的第 j 件物品的成本为 v[i][j]，价值为 w[i][j]。
// 每组有若干个物品，同一组内的物品最多只能选一个。
// 求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大。
func maxValue5(n, c int, s []int, v, w [][]int) int {
	dp := make([]int, c+1)
	for i := 0; i < n; i++ {
		si, vi, wi := s[i], v[i], w[i]
		for j := c; j >= 0; j-- {
			for k := 0; k < si; k++ {
				if j >= vi[k] {
					dp[j] = max(dp[j], dp[j-vi[k]]+wi[k])
				}
			}
		}
	}
	return dp[c]
}

// 分组背包例题——掷骰子（与模板不同的是每个组里必须选一个）
// n个骰子，每个骰子有k个面，掷骰子的得到总点数为各骰子面朝上的数字的总和。
// 如果需要掷出的总点数为 target，请你计算出有多少种不同的组合情况,最后模1e9+7返回
func numRollsToTarget(n int, k int, target int) int {
	const MOD int = 1e9 + 7
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 0; i < n; i++ {
		for j := target; j >= 0; j-- {
			// 由于我们直接是在 dp[i][j] 格子的基础上进行方案数累加，因此在计算 dp[i][j] 记得手动置零。
			dp[j] = 0
			for a := 1; a <= k && a <= j; a++ {
				dp[j] = (dp[j] + dp[j-a]) % MOD
			}
		}
	}
	return dp[target]
}

// 树形背包
// 有 n 个物品和一个容量为 c 的背包，物品编号为 0...N - 1。
// 物品之间具有依赖关系，且依赖关系组成一棵树的形状。
// 如果选择一个物品，则必须选择它的父节点。
// 第 i 件物品的体积为 v[i]，价值为 w[i]，其父节点物品编号为 p[i]，其中根节点 p[i] == -1。
// 求解将哪些物品装入背包，可使这些物品的总体积不超过背包容量，且总价值最大。
// dp[x][j]为考虑以 x 为根的子树，背包容量不超过 j 的最大价值。
// 可以根据「容量」这个维度对所有方案进行划分：
// 消耗容量为 0 的方案数的最大价值；
// 消耗容量为 1 的方案数的最大价值；
// ...
// 消耗容量为 j - v[x] 的方案数的最大价值；
// 消耗的容量的范围为 [0, j - v[x]]，是因为需要预留 v[x] 的容量选择当前的根节点 x 。
// 综上，最终的状态转移方程为（child 为节点 x 的子节点）：
// dp[x][j] = max(dp[x][j], dp[x][j-k]+dp[child][k])  0 <= k <= j - v[x]
// 从状态转移方式发现，在计算 f[x][j] 时需要用到 f[child][k]，因此我们需要先递归处理节点 x 的子节点 child 的状态值。
func maxValue6(n, c int, p, v, w []int) int {
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, c+1)
	}
	// 建图
	dic := make([][]int, n)
	root := -1
	for i, fa := range p {
		if fa == -1 {
			root = i
		} else {
			dic[fa] = append(dic[fa], i)
		}
	}
	var dfs func(int)
	dfs = func(x int) {
		vx, wx := v[x], w[x]
		// 要选任一节点，必须先选 x，同时也限制了至少需要 vx 的容量
		for i := vx; i <= c; i++ {
			dp[x][i] += wx
		}
		// 遍历节点 x 的所有子节点 child（分组背包遍历物品组）
		for _, child := range dic[x] {
			// 递归处理节点 child
			dfs(child)
			// 从大到小遍历背包容量（分组背包遍历容量）
			for j := c; j >= 0; j-- {
				// 遍历给节点 child 分配多少背包容量（分组背包遍历决策）
				for k := 0; k <= j-vx; k++ {
					dp[x][j] = max(dp[x][j], dp[x][j-k]+dp[child][k])
				}
			}
		}
	}
	dfs(root)
	return dp[root][c]
}

// 循环双端队列
// https://leetcode.cn/problems/design-circular-deque/
type MyCircularDeque struct {
	a               []int
	head, tail, cap int
}

func MyCircularDequeConstructor(k int) MyCircularDeque {
	return MyCircularDeque{make([]int, k+1), 0, k, k + 1}
}

func (this *MyCircularDeque) InsertFront(value int) bool {
	if this.IsFull() {
		return false
	}
	this.head = (this.head - 1 + this.cap) % this.cap
	this.a[this.head] = value
	return true
}

func (this *MyCircularDeque) InsertLast(value int) bool {
	if this.IsFull() {
		return false
	}
	this.tail = (this.tail + 1) % this.cap
	this.a[this.tail] = value
	return true
}

func (this *MyCircularDeque) DeleteFront() bool {
	if this.IsEmpty() {
		return false
	}
	this.head = (this.head + 1) % this.cap
	return true
}

func (this *MyCircularDeque) DeleteLast() bool {
	if this.IsEmpty() {
		return false
	}
	this.tail = (this.tail - 1 + this.cap) % this.cap
	return true
}

func (this *MyCircularDeque) GetFront() int {
	if this.IsEmpty() {
		return -1
	}
	return this.a[this.head]
}

func (this *MyCircularDeque) GetRear() int {
	if this.IsEmpty() {
		return -1
	}
	return this.a[this.tail]
}

func (this *MyCircularDeque) IsEmpty() bool {
	if (this.tail+1)%this.cap == this.head {
		return true
	}
	return false
}

func (this *MyCircularDeque) IsFull() bool {
	if (this.tail+2)%this.cap == this.head {
		return true
	}
	return false
}

// 子集dp
// 完成任务的最少工作时段
// https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/
func minSessions(tasks []int, sessionTime int) int {
	n := len(tasks)
	valid := make([]bool, 1<<n)
	for i := 0; i < (1 << n); i++ { // 预处理满足能在一个sessiontime里完成的任务组合
		c := 0
		for j := 0; j < n; j++ {
			if (i & (1 << j)) == 1<<j {
				c += tasks[j]
			}
		}
		if c <= sessionTime {
			valid[i] = true
		}
	}
	dp := make([]int, 1<<n)
	for i := 1; i < (1 << n); i++ {
		dp[i] = math.MaxInt / 2
	}
	for i := 1; i < (1 << n); i++ {
		// 枚举 i 的所有子集 s，若 s 耗时不超过 sessionTime，则将 dp[i^s]+1 转移到 dp[i] 上
		for s := i; s > 0; s = (s - 1) & i { // 遍历子集技巧
			if valid[s] {
				dp[i] = min(dp[i], dp[i^s]+1)
			}
		}
	}
	return dp[(1<<n)-1]
}

// 并行课程II 状压、子集dp
// https://leetcode.cn/problems/parallel-courses-ii/
// bits.Len() 计算二进制位数
// bits.TrailingZeros() 返回尾随零位的位数 注意 bits.TrailingZeros(0) 为uint的size，本机为64
// bits.OnesCount() 返回二进制中1的个数
func minNumberOfSemesters(n int, relations [][]int, k int) int {
	// 计算每个课程的先修课程
	pre := make([]int, n)
	for _, v := range relations {
		pre[v[1]-1] |= 1 << (v[0] - 1)
	}
	m := 1 << n
	// 计算一学期能修的合法课程的先修课程，不合法返回-1
	toPre := make([]int, m)
	for i := range toPre {
		if bits.OnesCount(uint(i)) > k {
			toPre[i] = -1
			continue
		}
		for j := i; j > 0; j &= j - 1 {
			p := pre[bits.TrailingZeros(uint(j))]
			if i&p > 0 {
				toPre[i] = -1
				break
			}
			toPre[i] |= p
		}
	}
	dp := make([]int, m)
	for i := range dp {
		dp[i] = n
	}
	dp[0] = 0
	for i, v := range dp {
		cp := (m - 1) ^ i                      // 补集
		for s := cp; s > 0; s = (s - 1) & cp { // 枚举下学期要上的课
			if p := toPre[s]; p != -1 && i&p == p { // 这些课的先修课必须合法且在之前的学期里必须上过
				dp[i|s] = min(dp[i|s], v+1)
			}
		}
	}
	return dp[m-1]
}

// 分配重复整数 状压 + 子集dp
// https://leetcode.cn/problems/distribute-repeating-integers/
func canDistribute(nums []int, quantity []int) bool {
	m := 1 << len(quantity)
	sum := make([]int, m)
	for i, v := range quantity { // 预处理 quantity 每个子集的子集和
		t := 1 << i
		for j := 0; j < t; j++ {
			sum[j|t] = sum[j] + v
		}
	}
	dic := map[int]int{}
	for _, v := range nums {
		dic[v]++
	}
	n := len(dic)
	// dp[i][j] 表示 dic 的前 i 个元素能否满足集合为 j 的顾客
	dp := make([][]bool, n+1)
	for i := range dp {
		dp[i] = make([]bool, m)
		dp[i][0] = true
	}
	i := 0 // 将外层循环即dic与dp的第一维对应起来
	for _, c := range dic {
		for j, ok := range dp[i] {
			if ok {
				dp[i+1][j] = true
				continue
			} else {
				for s := j; s > 0; s = (s - 1) & j { // 枚举 j 的子集 s
					if sum[s] <= c && dp[i][j^s] { // 判断这 c 个数能否全部分给 sub，并且除了 sub 以外的 j 中的顾客也满足
						dp[i+1][j] = true
						break
					}
				}
			}
		}
		i++
	}
	return dp[n][m-1]
}

// 乘积小于K的子数组  滑动窗口
// https://leetcode.cn/problems/subarray-product-less-than-k/
func numSubarrayProductLessThanK(nums []int, k int) (ans int) {
	if k == 0 {
		return 0
	}
	l, r, n, cur := 0, 0, len(nums), 1
	for r < n {
		cur *= nums[r]
		r++
		for cur >= k && l < r {
			cur /= nums[l]
			l++
		}
		ans += (r - l) // 加上以nums[r-1]结尾的所有符合要求的子数组个数
	}
	return
}

// 0 和 1 个数相同的子数组   前缀和 + 哈希
// https://leetcode.cn/problems/A1NYOS/
func findMaxLength(nums []int) (ans int) {
	dic := map[int]int{}
	dic[0] = 0
	s := 0
	for i, v := range nums {
		s += v*2 - 1
		if l, ok := dic[s]; ok {
			ans = max(ans, i+1-l)
		} else {
			dic[s] = i + 1
		}
	}
	return
}

// 前缀和 + 哈希 其他题目：和为 k 的子数组  https://leetcode.cn/problems/subarray-sum-equals-k/

// 最大价值和与最小价值和的差值   树形dp  dfs多返回值  转换成树上最大路径和
// https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/
func maxOutput(n int, edges [][]int, price []int) int64 {
	g := make([][]int, n)
	for _, e := range edges {
		g[e[0]] = append(g[e[0]], e[1])
		g[e[1]] = append(g[e[1]], e[0])
	}
	ans := 0
	var dfs func(int, int) (int, int) // 返回带端点的最长路径和不带端点的最长路径
	dfs = func(x, fa int) (int, int) {
		ma, mb := price[x], 0 // 带端点的最长路径和不带端点的最长路径
		for _, c := range g[x] {
			if c == fa {
				continue
			}
			a, b := dfs(c, x)
			ans = max(ans, max(ma+b, mb+a)) // 先更新答案再更新ma, mb 就不会重复
			ma = max(ma, a+price[x])
			mb = max(mb, b+price[x])
		}
		return ma, mb
	}
	dfs(0, -1)
	return int64(ans)
}

// 回文链表   findMid reverse 的写法  条件：链表节点数至少为1
// https://leetcode.cn/problems/palindrome-linked-list/
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
type ListNode struct {
	Val  int
	Next *ListNode
}

func isPalindrome(head *ListNode) bool {
	findMid := func(r *ListNode) *ListNode {
		low, fast := r, r
		for fast.Next != nil && fast.Next.Next != nil {
			low = low.Next
			fast = fast.Next.Next
		}
		return low
	}
	reverse := func(r *ListNode) *ListNode {
		dh := &ListNode{}
		for r != nil {
			t := r.Next
			r.Next = dh.Next
			dh.Next = r
			r = t
		}
		return dh.Next
	}
	juj := func(a, b *ListNode) bool {
		for a != nil && b != nil {
			if a.Val != b.Val {
				return false
			}
			a = a.Next
			b = b.Next
		}
		return true
	}
	mid := findMid(head)
	b := reverse(mid.Next)
	mid.Next = nil // 注意切割
	return juj(head, b)
}

// 到达角落需要移除障碍物的最小数目   0-1 bfs
// https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/
// 0-1 bfs 适合边权为0或者为1的情况，如果边权大于1直接上堆
// 0-1 bfs 其他例题：
// 使网格图至少有一条有效路径的最小代价  https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/
// 最少侧跳次数  https://leetcode.cn/problems/minimum-sideway-jumps/
func minimumObstacles(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	d := make([][]int, m)
	for i := range d {
		d[i] = make([]int, n)
		for j := range d[i] {
			d[i][j] = m * n
		}
	}
	dir := []int{-1, 0, 1, 0, -1}
	l, r := [][2]int{}, [][2]int{{0, 0}} // 两个 slice 头对头来实现 deque
	d[0][0] = 0
	for {
		t := [2]int{}
		if len(l) > 0 {
			t, l = l[len(l)-1], l[:len(l)-1]
		} else {
			t, r = r[0], r[1:]
		}
		x, y := t[0], t[1]
		if x == m-1 && y == n-1 {
			return d[x][y]
		}
		for i := 1; i < 5; i++ {
			nx, ny := x+dir[i-1], y+dir[i]
			if nx >= 0 && nx < m && ny >= 0 && ny < n {
				if grid[nx][ny] == 1 && d[x][y]+1 < d[nx][ny] {
					d[nx][ny] = d[x][y] + 1
					r = append(r, [2]int{nx, ny}) // 加到队尾
				} else if grid[nx][ny] == 0 && d[x][y] < d[nx][ny] {
					d[nx][ny] = d[x][y]
					l = append(l, [2]int{nx, ny}) // 加到队首
				}
			}
		}
	}
}

// 判断一个点是否可以到达    判断gcd是否是2的幂次
// (x, y - x) (x - y, y)联想到gcd
// https://leetcode.cn/problems/check-if-point-is-reachable/
func isReachable(x int, y int) bool {
	// 法一：构造
	// x y 任一为偶数则除以2
	// 都是奇数时，将较大的奇数变为(x + y) / 2
	// 都是奇数且x == y 时，若都为1返回true 否则返回false
	// f := func() {
	//     for x & 1 == 0 || y & 1 == 0 {
	//         if x & 1 == 0 {
	//             x >>= 1
	//         }
	//         if y & 1 == 0 {
	//             y >>= 1
	//         }
	//     }
	// }
	// f()
	// for x != y {
	//     f()
	//     if x > y {
	//         x = (x + y) / 2
	//     } else if x < y {
	//         y = (x + y) / 2
	//     }

	// }
	// return x == 1

	// 法二：判断gcd是否是二的幂次
	var gcd func(int, int) int
	gcd = func(x, y int) int {
		if x%y == 0 {
			return y
		}
		return gcd(y, x%y)
	}
	t := gcd(x, y)
	return t&(t-1) == 0
}

// 往完全二叉树添加节点   完全二叉树dfs与二进制表示的关联
// https://leetcode.cn/problems/NaqhDT/
type CBTInserter struct {
	root *TreeNode
	cnt  int
}

func CBTConstructor(root *TreeNode) CBTInserter {
	var dfs func(*TreeNode) int
	dfs = func(r *TreeNode) int {
		if r == nil {
			return 0
		}
		return dfs(r.Left) + dfs(r.Right) + 1
	}
	return CBTInserter{root, dfs(root)}
}

func (this *CBTInserter) Insert(v int) int {
	tnode := &TreeNode{Val: v}
	this.cnt++
	node, c := this.root, this.cnt
	for i := bits.Len(uint(c)) - 2; i > 0; i-- {
		if c>>i&1 > 0 {
			node = node.Right
		} else {
			node = node.Left
		}
	}
	if c&1 > 0 {
		node.Right = tnode
	} else {
		node.Left = tnode
	}
	return node.Val
}

func (this *CBTInserter) Get_root() *TreeNode {
	return this.root
}

// 序列化与反序列化二叉树   dfs
// https://leetcode.cn/problems/h54YBf/
type Codec struct {
}

func serializeConstructor() (_ Codec) {
	return
}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	var sb strings.Builder
	var dfs func(*TreeNode)
	dfs = func(root *TreeNode) {
		if root == nil {
			sb.WriteString("nil,")
			return
		}
		sb.WriteString(strconv.Itoa(root.Val))
		sb.WriteByte(',')
		dfs(root.Left)
		dfs(root.Right)
	}
	dfs(root)
	return sb.String()
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
	a := strings.Split(data, ",")
	var build func() *TreeNode
	build = func() *TreeNode {
		if a[0] == "nil" {
			a = a[1:]
			return nil
		}
		num, _ := strconv.Atoi(a[0])
		a = a[1:]
		return &TreeNode{num, build(), build()}
	}
	return build()
}

// 值和下标之差都在给定的范围内   桶，一个桶只有一个元素所以用哈希
// https://leetcode.cn/problems/contains-duplicate-iii/
func containsNearbyAlmostDuplicate(nums []int, k int, t int) bool {
	getID := func(x, t int) int {
		if x >= 0 {
			return x / t
		}
		return (x+1)/t - 1
	}
	dic := map[int]int{}
	for i, v := range nums {
		id := getID(v, t+1)
		if _, ok := dic[id]; ok {
			return true
		}
		if tv, ok := dic[id-1]; ok && abs(tv-v) <= t {
			return true
		}
		if tv, ok := dic[id+1]; ok && abs(tv-v) <= t {
			return true
		}
		dic[id] = v
		if i >= k {
			delete(dic, getID(nums[i-k], t+1))
		}
	}
	return false
}
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// 二进制数转字符串   任何进制表示的小数，乘上进制等价于小数点往右移一位
// https://leetcode.cn/problems/bianry-number-to-string-lcci/
func printBin(num float64) string {
	ans := &strings.Builder{}
	ans.WriteString("0.")
	for ans.Len() <= 32 && num != 0 {
		num *= 2
		t := int(num)
		ans.WriteByte(byte('0' + t))
		num -= float64(t)
	}
	if ans.Len() <= 32 {
		return ans.String()
	}
	return "ERROR"
}
