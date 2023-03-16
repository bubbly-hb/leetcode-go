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

	fmt.Fprintln(out, n)
}
