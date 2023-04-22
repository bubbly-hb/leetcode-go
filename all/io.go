package all

import (
	"bufio"
	"fmt"
	"os"
)

func bufferIO() {
	in := bufio.NewReader(os.Stdin)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	var n int
	fmt.Fscan(in, &n) // 如果行数未知，可以根据 Fscan 的第一个返回值是否为正来决定

	// s, _ := in.ReadString('\n')  // 读入一行未知个数的整数数组
	// ss := strings.Fields(s)
	// a := make([]int, len(ss))
	// for i, v := range ss {
	// 	a[i], _ = strconv.Atoi(v)
	// }

	fmt.Fprintln(out, n)
}
