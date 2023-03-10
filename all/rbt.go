package all

import (
	"math"

	"github.com/Arafatk/DataViz/trees/redblacktree"
)

// redblacktree
// 不支持重复key
// redblacktree 维护了一个(key, value)集合，Put根据key来覆盖或者新增，Get根据key返回value(interface{}), bool
// Left().Key 返回最小key   Right().Key 返回最大key  根据情况加断言
// Empty() 为空返回true
// Dataviz是一个数据结构可视化库，通过对 Graphviz的可视化功能用golang来重新打包，已实现在golang中实现基本数据结构的可视化。
// 实现红黑树的地址：https://github.com/Arafatk/DataViz/tree/master/trees/redblacktree
/*
（1）红黑树的树中结点：redblacktree.Node。在源码中，结构体Node内部包含存放的结点值Key、Key对应的Value、代表的颜色color、
	结点索引nodeIndex、左右节点Left或Right、父节点Parent等。

（2）红黑树对象类型：redblacktree.Tree。在源码中，结构体Tree内部包含一个类型为*Node的变量Root、大小size、以及一个用于给
	Tree中的Key排序的Comparator（比较器）变量。

（3）Comparator类型：func(o1, o2 interface{}) int。在源码中，type Comparator被定义为上述的函数结构。具体的Comparator包括
	IntComparator和StringComparator。

（4）创建红黑树对象：

　　① redblacktree.NewWith(c Comparator)：该方法需要传入一个Comparator对象。

　　② redblacktree.NewWithIntComparator() / redblacktree.NewWithStringComparator() ：这类方法可通过特定的Comparator创建
	红黑树对象。

（5）常用操作：（为便于演示，此处创建Tree对象：tree := redblacktree.NewTree()）

　　① 根据key值获取value值：val, b := tree.Get(key)。返回值b为bool类型，表示获取成功或失败；val为tree中结点值为key对应的value值。

　　② 加入键值对key-value：tree.Put(key, value)。

　　③ 获取tree中的结点数目：tree.Size()。返回值为int型。

　　④ 根据key删除某结点：tree.Remove(key)。

　　⑤ 获取tree中最小/最大的key对应的结点：tree.Left()。返回值为Node类型，通过tree.Left().Key可得到该Node对象存放的key
	值（interface{}类型），若此前定义过Comparator的类型为int或string，可通过断言获取实际key值：如通过tree.Left().Key.(int)可
	获取到类型为int的key值，而非interface{}。Left()获取最小key值对应的Node，Right()获取最大key值对应的Node。

　　⑥ 获取tree中小于等于key的最大key对应的Node：floor, b := tree.Floor(key)。floor为*Node类型，b为bool类型。

　　⑦ 获取tree中大于等于key的最小key对应的Node：ceiling, b := tree.Ceiling(key)。ceiling为*Node类型，b为bool类型。
*/

// 股票价格波动
// https://leetcode.cn/problems/stock-price-fluctuation/
type StockPrice struct {
	prices *redblacktree.Tree // key : 价格， val : 价格为key的时间戳的数量
	dic    map[int]int        // 时间戳对应的价格
	maxts  int
}

func ConstructorStock() StockPrice {
	return StockPrice{redblacktree.NewWithIntComparator(), map[int]int{}, 0}
}

func (this *StockPrice) Update(timestamp int, price int) {
	if p := this.dic[timestamp]; p > 0 {
		if c, _ := this.prices.Get(p); c.(int) > 1 {
			this.prices.Put(p, c.(int)-1)
		} else {
			this.prices.Remove(p)
		}
	}
	c := 0
	if tc, ok := this.prices.Get(price); ok {
		c = tc.(int)
	}
	this.prices.Put(price, c+1)
	this.dic[timestamp] = price
	if timestamp > this.maxts {
		this.maxts = timestamp
	}
}

func (this *StockPrice) Current() int {
	return this.dic[this.maxts]
}

func (this *StockPrice) Maximum() int {
	return this.prices.Right().Key.(int)
}

func (this *StockPrice) Minimum() int {
	return this.prices.Left().Key.(int)
}

// 考场就坐  自定义comp函数，红黑树的key为[]int（二元组l, r），值不关心所以设为任意struct{}{}
// 考虑到每次 seat() 时都需要找到最大距离的座位，我们可以使用有序集合来保存座位区间。有序集合的每个元素为一个二元组 (l, r)，表示 l 和 r 之间
//（不包括 l 和 r）的座位可以坐学生。初始时有序集合中只有一个元素 (-1, n)，表示 (-1, n) 之间的座位可以坐学生。
// 另外，我们使用两个哈希表 left 和 right 来维护每个有学生的座位的左右邻居学生，方便我们在 leave(p) 时合并两个座位区间。
// https://leetcode.cn/problems/exam-room/
type ExamRoom struct {
	rbt   *redblacktree.Tree
	left  map[int]int
	right map[int]int
	n     int
}

func ExamRoomConstructor(n int) ExamRoom {
	dist := func(a []int) int {
		if a[0] == -1 || a[1] == n {
			return a[1] - a[0] - 1
		}
		return (a[1] - a[0]) >> 1
	}
	comp := func(a, b interface{}) int { // a比b优先级高时返回正数，这样的话.Right().Key取到的就是高优先级的key
		x, y := a.([]int), b.([]int)
		d1, d2 := dist(x), dist(y)
		if d1 == d2 {
			return y[0] - x[0]
		}
		return d1 - d2
	}
	this := ExamRoom{redblacktree.NewWith(comp), map[int]int{}, map[int]int{}, n}
	this.add([]int{-1, n})
	return this
}

func (this *ExamRoom) Seat() int {
	a := this.rbt.Right().Key.([]int)
	d := (a[1] + a[0]) >> 1
	if a[0] == -1 {
		d = 0
	} else if a[1] == this.n {
		d = this.n - 1
	}
	this.del(a) // 注意这里是先删后加，不然先加的话再删除时会改变left, right哈希值
	this.add([]int{a[0], d})
	this.add([]int{d, a[1]})
	return d
}

func (this *ExamRoom) Leave(p int) {
	l, r := this.left[p], this.right[p]
	this.del([]int{l, p})
	this.del([]int{p, r})
	this.add([]int{l, r})
}

func (this *ExamRoom) add(a []int) {
	this.rbt.Put(a, struct{}{})
	this.left[a[1]] = a[0]
	this.right[a[0]] = a[1]
}

func (this *ExamRoom) del(a []int) {
	this.rbt.Remove(a)
	delete(this.left, a[1])
	delete(this.right, a[0])
}

// 求出 MK 平均值
// https://leetcode.cn/problems/finding-mk-average/
type MKAverage struct {
	l, mid, r         *redblacktree.Tree
	q                 []int
	m, k, sum, nl, nr int // sum 为 mid 中所有元素和，nl, nr 分别为l, r中元素数量
}

func MKAverageConstructor(m int, k int) MKAverage {
	l := redblacktree.NewWithIntComparator()
	mid := redblacktree.NewWithIntComparator()
	r := redblacktree.NewWithIntComparator()
	return MKAverage{l, mid, r, []int{}, m, k, 0, 0, 0}
}

func merge(r *redblacktree.Tree, key, val int) { // 将添加与删除统一起来
	if tc, ok := r.Get(key); ok {
		val += tc.(int)
		if val == 0 {
			r.Remove(key)
		} else {
			r.Put(key, val)
		}
	} else {
		r.Put(key, val)
	}
}

// 得先添加再删除，比如m == 3, k == 1, 加入3， 4， 5后加入6时如果先删除3, 6会被直接加到l
// 但是如果先加6, 6会被加到r, 然后再删除3，再进行l, mid, r之间的调整
func (this *MKAverage) AddElement(num int) {
	this.q = append(this.q, num)
	// 添加num
	if this.nl == 0 || num < this.l.Right().Key.(int) {
		merge(this.l, num, 1)
		this.nl++
	} else if this.nr == 0 || num > this.r.Left().Key.(int) {
		merge(this.r, num, 1)
		this.nr++
	} else {
		merge(this.mid, num, 1)
		this.sum += num
	}
	// 删除多余的元素
	if len(this.q) > this.m {
		t := this.q[0]
		this.q = this.q[1:]
		if _, ok := this.l.Get(t); ok {
			merge(this.l, t, -1)
			this.nl--
		} else if _, ok := this.r.Get(t); ok {
			merge(this.r, t, -1)
			this.nr--
		} else {
			merge(this.mid, t, -1)
			this.sum -= t
		}
	}
	// 调整 l, mid, r 内的元素
	for this.nl > this.k {
		t := this.l.Right().Key.(int)
		merge(this.l, t, -1)
		merge(this.mid, t, 1)
		this.nl--
		this.sum += t
	}
	for this.nr > this.k {
		t := this.r.Left().Key.(int)
		merge(this.r, t, -1)
		merge(this.mid, t, 1)
		this.nr--
		this.sum += t
	}
	for this.nl < this.k && !this.mid.Empty() {
		t := this.mid.Left().Key.(int)
		merge(this.mid, t, -1)
		merge(this.l, t, 1)
		this.nl++
		this.sum -= t
	}
	for this.nr < this.k && !this.mid.Empty() {
		t := this.mid.Right().Key.(int)
		merge(this.mid, t, -1)
		merge(this.r, t, 1)
		this.nr++
		this.sum -= t
	}
}

func (this *MKAverage) CalculateMKAverage() int {
	if len(this.q) < this.m {
		return -1
	}
	return this.sum / (this.m - this.k*2)
}

/**
 * Your MKAverage object will be instantiated and called as such:
 * obj := Constructor(m, k);
 * obj.AddElement(num);
 * param_2 := obj.CalculateMKAverage();
 */

// 日程表    Prev() Next() 会修改迭代器的值并返回bool
// https://leetcode.cn/problems/my-calendar-i/
type MyCalendar struct {
	*redblacktree.Tree
}

func MyCalendarConstructor() MyCalendar {
	t := redblacktree.NewWithIntComparator()
	t.Put(math.MaxInt32, nil) // 哨兵，简化代码
	return MyCalendar{t}
}

// func (c MyCalendar) Book(start, end int) bool {
// 	node, _ := c.Ceiling(end)
// 	it := c.IteratorAt(node)
// 	if !it.Prev() || it.Value().(int) <= start {    // it.Prev()已经将迭代器指向上一个元素了
// 		c.Put(start, end)
// 		return true
// 	}
// 	return false
// }
