package all

type seg []struct{ l, r, max int }

func (s seg) build(o, l, r int) {
	s[o].l, s[o].r = l, r
	if l == r { // 记得return
		return
	}
	m := (l + r) >> 1
	s.build(o<<1, l, m)
	s.build(o<<1|1, m+1, r)
}
func (s seg) modify(o, i, val int) {
	if s[o].l == s[o].r {
		s[o].max = val
		return
	}
	m := (s[o].l + s[o].r) >> 1
	if i <= m {
		s.modify(o<<1, i, val)
	} else {
		s.modify(o<<1|1, i, val)
	}
	s[o].max = max(s[o<<1].max, s[o<<1|1].max)
}
func (s seg) query(o, l, r int) int {
	if l <= s[o].l && r >= s[o].r {
		return s[o].max
	}
	m := (s[o].l + s[o].r) >> 1
	if r <= m {
		return s.query(o<<1, l, r)
	}
	if l > m {
		return s.query(o<<1|1, l, r)
	}
	return max(s.query(o<<1, l, r), s.query(o<<1|1, l, r))
}

// 最长递增子序列II  https://leetcode.cn/problems/longest-increasing-subsequence-ii/  线段树优化dp

/*
// 更新数组后处理求和查询  lazy线段树
// https://leetcode.cn/problems/handling-sum-queries-after-update/
func handleQuery(nums1 []int, nums2 []int, queries [][]int) (ans []int64) {
	sum := 0
	for _, v := range nums2 {
		sum += v
	}
	n := len(nums1)
	s := make(seg, n*4)
	s.build(nums1, 1, 1, n)
	for _, q := range queries {
		if q[0] == 1 {
			s.update(1, q[1]+1, q[2]+1)
		} else if q[0] == 2 {
			sum += s[1].cnt1 * q[1]
		} else {
			ans = append(ans, int64(sum))
		}
	}
	return
}

type seg []struct {
	l, r, cnt1 int
	flip       bool
}

func (s seg) maintain(o int) {
	s[o].cnt1 = s[o<<1].cnt1 + s[o<<1|1].cnt1
}
func (s seg) build(a []int, o, l, r int) {
	s[o].l, s[o].r = l, r
	if l == r {
		s[o].cnt1 = a[l-1]
		return
	}
	m := (l + r) >> 1
	s.build(a, o<<1, l, m)   // 这里是l, m , 不要写成l, r
	s.build(a, o<<1|1, m+1, r) // 同上，不要顺手写成l, r
	s.maintain(o)
}
func (s seg) do(o int) {
	s[o].cnt1 = s[o].r - s[o].l + 1 - s[o].cnt1
	s[o].flip = !s[o].flip
}
func (s seg) spread(o int) {
	if s[o].flip {
		s.do(o << 1)
		s.do(o<<1 | 1)
		s[o].flip = false
	}
}
func (s seg) update(o, l, r int) {
	if l <= s[o].l && s[o].r <= r {
		s.do(o)
		return
	}
	s.spread(o)
	m := (s[o].l + s[o].r) >> 1  // 这里是s[o].l 与 s[o].r 不要顺手写成l, r
	if l <= m {
		s.update(o<<1, l, r)
	}
	if r > m {
		s.update(o<<1|1, l, r)
	}
	s.maintain(o)
}
*/
