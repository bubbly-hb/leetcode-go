package all

// 并查集模板
type unionFind struct {
	parent, size []int
}

func newUnionFind(n int) *unionFind {
	parent, size := make([]int, n), make([]int, n)
	for i := 0; i < n; i++ {
		parent[i] = i
		size[i] = 1
	}
	return &unionFind{parent, size}
}
func (uf *unionFind) find(x int) int {
	if x != uf.parent[x] {
		uf.parent[x] = uf.find(uf.parent[x])
	}
	return uf.parent[x]
}
func (uf *unionFind) union(x, y int) {
	fx, fy := uf.find(x), uf.find(y)
	if fx == fy {
		return
	}
	if uf.size[fx] < uf.size[fy] {
		fx, fy = fy, fx
	}
	uf.size[fx] += uf.size[fy]
	uf.parent[fy] = fx
}
func (uf *unionFind) inSame(x, y int) bool {
	return uf.find(x) == uf.find(y)
}

// 并查集例题：
// 好路径的数目 https://leetcode.cn/problems/number-of-good-paths/

// 计算除法 带权值的并查集
// https://leetcode.cn/problems/vlzXQL/
// type unionFind struct {
// 	parent []int
// 	val    []float64
// }

// func newUnionFind(n int) *unionFind {
// 	parent, val := make([]int, n), make([]float64, n)
// 	for i := 0; i < n; i++ {
// 		parent[i] = i
// 		val[i] = 1
// 	}
// 	return &unionFind{parent, val}
// }
// func (uf *unionFind) find(x int) int {
// 	if x != uf.parent[x] {
// 		op := uf.parent[x]
// 		uf.parent[x] = uf.find(uf.parent[x])
// 		uf.val[x] *= uf.val[op]
// 	}
// 	return uf.parent[x]
// }
// func (uf *unionFind) union(x, y int, val float64) {    // x == y * val
// 	fx, fy := uf.find(x), uf.find(y)
// 	if fx == fy {
// 		return
// 	}
// 	uf.parent[fx] = fy
// 	uf.val[fx] = val * uf.val[y] / uf.val[x]
// }
// func (uf *unionFind) inSame(x, y int) bool {
// 	return uf.find(x) == uf.find(y)
// }
