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
