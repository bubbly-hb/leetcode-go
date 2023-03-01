package all

type MqData struct {
	Val, Del int
}
type MonotoneQueue struct {
	Data []*MqData
	Size int
	Tag  bool // false: Less 为 <= 维护区间最小值;  true: Less 为 >= 维护区间最大值
}

func (mq MonotoneQueue) Less(a, b *MqData) bool { // >= 维护区间最大值 <= 维护区间最小值
	if mq.Tag {
		return a.Val >= b.Val
	}
	return a.Val <= b.Val
}
func (mq *MonotoneQueue) Push(x int) {
	d := &MqData{x, 1}
	for len(mq.Data) > 0 && mq.Less(d, mq.Data[len(mq.Data)-1]) {
		d.Del += mq.Data[len(mq.Data)-1].Del
		mq.Data = mq.Data[:len(mq.Data)-1]
	}
	mq.Data = append(mq.Data, d)
	mq.Size++
}
func (mq *MonotoneQueue) Pop() {
	if mq.Data[0].Del == 1 {
		mq.Data = mq.Data[1:]
	} else {
		mq.Data[0].Del--
	}
	mq.Size--
}
func (mq *MonotoneQueue) Top() int {
	return mq.Data[0].Val
}

// 二维单调队列
// 输入：一个 m 行 n 列的矩阵 mat
// 输入：高 h 宽 w 的窗口大小
// 返回：一个 m-h+1 行 n-w+1 列的矩阵 areaMax，其中 areaMax[i][j] 表示窗口左上角位于矩阵 (i,j) 时的窗口中元素的最值 tag为true时返回最大值，否则返回最小值
// 例题：HA07 理想的正方形 https://www.luogu.com.cn/problem/P2216
// 解释：https://cdn.acwing.com/media/article/image/2021/06/29/52559_7d7b27ced8-1.png
func FixedSizeAreaMx(mat [][]int, h, w int, tag bool) [][]int {
	m, n := len(mat), len(mat[0])
	// 先对每行横向滑预处理，连续w个值的最大值保存到最右边的那个元素
	a := make([][]int, m)
	for i := range a {
		a[i] = make([]int, n-w+1)
	}
	for i, row := range mat {
		mq := &MonotoneQueue{Tag: tag}
		for j, v := range row {
			mq.Push(v)
			if mq.Size > w {
				mq.Pop()
			}
			if j+1 >= w {
				a[i][j+1-w] = mq.Top()
			}
		}
	}
	areaMax := make([][]int, m-h+1)
	for i := range areaMax {
		areaMax[i] = make([]int, n-w+1)
	}
	for j := 0; j < n-w+1; j++ {
		mq := &MonotoneQueue{Tag: tag}
		for i := 0; i < m; i++ {
			mq.Push(a[i][j])
			if mq.Size > h {
				mq.Pop()
			}
			if i+1 >= h {
				areaMax[i+1-h][j] = mq.Top()
			}
		}
	}
	return areaMax
}
