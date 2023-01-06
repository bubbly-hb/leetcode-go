package all

import "sort"

// 树状数组  [1, n]
// tree := make([]int, n+1)
// add := func(x, val int) {
// 	for ; x <= n; x += x & -x {
// 		tree[x] += val
// 	}
// }
// sum := func(x int) (ans int) {
// 	for ; x > 0; x &= x - 1 {
// 		ans += tree[x]
// 	}
// 	return
// }
// query := func(l, r int) int {
// 	return sum(r) - sum(l-1)
// }

// 满足不等式的数对数目
// https://leetcode.cn/problems/number-of-pairs-satisfying-ineuality/
func numberOfPairs(nums1 []int, nums2 []int, diff int) int64 {
	n := len(nums1)
	tree := make([]int, n+1)
	add := func(x, val int) {
		for ; x <= n; x += x & -x {
			tree[x] += val
		}
	}
	sum := func(x int) (ans int) {
		for ; x > 0; x &= x - 1 {
			ans += tree[x]
		}
		return
	}
	// query := func(l, r int) int {
	//     return sum(r) - sum(l - 1)
	// }
	a, b := make([]int, n), make([]int, n)
	for i := range a {
		a[i] = nums1[i] - nums2[i]
		b[i] = nums1[i] - nums2[i]
	}
	sort.Ints(b)
	ans := 0
	for _, v := range a {
		i := sort.SearchInts(b, v+diff+1) // 离散化 + 二分
		ans += sum(i)
		add(sort.SearchInts(b, v+1), 1)
	}
	return int64(ans)
}
