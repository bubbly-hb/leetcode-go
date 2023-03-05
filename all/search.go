package all

import "sort"

// 回溯
// 允许重复选择元素的组合  https://leetcode.cn/problems/Ygoe9J/
func combinationSum(candidates []int, target int) (ans [][]int) {
	a, n := []int{}, len(candidates)
	var dfs func(int, int)
	dfs = func(k, sum int) {
		if sum >= target || k == n {
			if sum == target {
				ans = append(ans, append([]int(nil), a...))
			}
			return
		}
		a = append(a, candidates[k])
		dfs(k, sum+candidates[k])
		a = a[:len(a)-1]
		dfs(k+1, sum)
	}
	dfs(0, 0)
	return
}

// 含有重复元素集合的组合  https://leetcode.cn/problems/4sjJUc/
func combinationSum2_(candidates []int, target int) (ans [][]int) {
	a, n := []int{}, len(candidates)
	sort.Ints(candidates)
	var dfs func(int, int, bool)
	dfs = func(x, sum int, preChoose bool) {
		if sum >= target || x == n {
			if sum == target {
				ans = append(ans, append([]int(nil), a...))
			}
			return
		}
		dfs(x+1, sum, false)
		if !preChoose && x > 0 && candidates[x] == candidates[x-1] {
			return
		}
		a = append(a, candidates[x])
		dfs(x+1, sum+candidates[x], true)
		a = a[:len(a)-1]
	}
	dfs(0, 0, false)
	return
}

// 没有重复元素集合的全排列   https://leetcode.cn/problems/VvJkup/
func permute(nums []int) (ans [][]int) {
	a := []int(nil)
	n := len(nums)
	dic := make([]bool, n)
	var dfs func(x int)
	dfs = func(x int) {
		if x == n {
			ans = append(ans, append([]int(nil), a...))
			return
		}
		for i, v := range nums {
			if !dic[i] {
				dic[i] = true
				a = append(a, v)
				dfs(x + 1)
				a = a[:len(a)-1]
				dic[i] = false
			}
		}
	}
	dfs(0)
	return
}

// 含有重复元素集合的全排列  https://leetcode.cn/problems/7p8L0Z/
func permuteUnique2(nums []int) (ans [][]int) {
	a, n := []int(nil), len(nums)
	sort.Ints(nums)
	dic := make([]bool, n)
	var dfs func(int)
	dfs = func(x int) {
		if x == n {
			ans = append(ans, append([]int(nil), a...))
			return
		}
		pre := -1
		for i, v := range nums {
			if !dic[i] && (pre == -1 || v != nums[pre]) {
				dic[i] = true
				a = append(a, v)
				pre = i
				dfs(x + 1)
				a = a[:len(a)-1]
				dic[i] = false
			}
		}
	}
	dfs(0)
	return
}
