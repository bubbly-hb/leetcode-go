package all

import "container/heap"

// 支持修改、删除指定元素的堆
// 用法：调用 push 会返回一个 *viPair 指针，记作 p
// 将 p 存于他处（如 slice 或 map），可直接在外部修改 p.v 后调用 fix(p.index)，从而做到修改堆中指定元素
// 调用 remove(p.index) 可以从堆中删除 p.v
// 例题 https://atcoder.jp/contests/abc170/tasks/abc170_e
// 模拟 multiset https://codeforces.com/problemset/problem/1106/E
// type viPair struct {
// 	v  int64
// 	hi int // *viPair 在 mh 中的下标，可随着 Push Pop 等操作自动改变
// }
// type mh []*viPair // mh 指 modifiable heap

// func (h mh) Len() int              { return len(h) }
// func (h mh) Less(i, j int) bool    { return h[i].v < h[j].v } // > 为最大堆
// func (h mh) Swap(i, j int)         { h[i], h[j] = h[j], h[i]; h[i].hi = i; h[j].hi = j }
// func (h *mh) Push(v interface{})   { *h = append(*h, v.(*viPair)) }
// func (h *mh) Pop() interface{}     { a := *h; v := a[len(a)-1]; *h = a[:len(a)-1]; return v }
// func (h *mh) push(v int64) *viPair { p := &viPair{v, len(*h)}; heap.Push(h, p); return p }
// func (h *mh) pop() *viPair         { return heap.Pop(h).(*viPair) }
// func (h *mh) fix(i int)            { heap.Fix(h, i) }
// func (h *mh) remove(i int) *viPair { return heap.Remove(h, i).(*viPair) }
func (h mh) replace(v int) int { // 用v替换堆顶并且返回原来的堆顶值
	top := h[0].v
	h[0].v = v
	heap.Fix(&h, 0)
	return top
}

type viPair struct {
	v  int
	hi int // *viPair 在 mh 中的下标，可随着 Push Pop 等操作自动改变
}
type mh []*viPair // mh 指 modifiable heap

type dic map[int][]*viPair

func (h mh) Len() int             { return len(h) }
func (h mh) Less(i, j int) bool   { return h[i].v < h[j].v } // > 为最大堆
func (h mh) Swap(i, j int)        { h[i], h[j] = h[j], h[i]; h[i].hi = i; h[j].hi = j }
func (h *mh) Push(v interface{})  { *h = append(*h, v.(*viPair)) }
func (h *mh) Pop() interface{}    { a := *h; v := a[len(a)-1]; *h = a[:len(a)-1]; return v }
func (h *mh) push(v int, d dic)   { p := &viPair{v, len(*h)}; heap.Push(h, p); d[v] = append(d[v], p) }
func (h *mh) pop(d dic) *viPair   { p := heap.Pop(h).(*viPair); d[p.v] = d[p.v][1:]; return p }
func (h *mh) fix(i int)           { heap.Fix(h, i) }
func (h *mh) remove(v int, d dic) { p := d[v][0]; d[v] = d[v][1:]; heap.Remove(h, p.hi) }

// 用到remove的例题：
// 天际线问题   https://leetcode.cn/problems/the-skyline-problem/

// 若只用到int的小根堆，可以用如下简洁的写法
// type mh struct{ sort.IntSlice }

// func (h *mh) Push(x any) { h.IntSlice = append(h.IntSlice, x.(int)) }
// func (h *mh) Pop() any {
// 	v := (h.IntSlice)[h.Len()-1]
// 	h.IntSlice = (h.IntSlice)[:h.Len()-1]
// 	return v
// }
