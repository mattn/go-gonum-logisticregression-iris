package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func logisticRegression(X []*mat.VecDense, y *mat.VecDense, rate float64, ntrains int) *mat.VecDense {
	ws := make([]float64, X[0].Len())
	for i := range ws {
		ws[i] = (rand.Float64() - 0.5) * float64(X[0].Len()/2)
	}
	w := mat.NewVecDense(len(ws), ws)
	for n := 0; n < ntrains; n++ {
		for i, x := range X {
			t := mat.NewVecDense(x.Len(), nil)
			t.CopyVec(x)
			pred := softmax(w, x)
			perr := y.At(i, 0) - pred
			scale := rate * perr * pred * (1 - pred)
			dx := mat.NewVecDense(x.Len(), nil)
			dx.CopyVec(x)
			dx.ScaleVec(scale, x)
			for j := 0; j < x.Len(); j++ {
				w.AddVec(w, dx)
			}
		}
	}
	return w
}

func softmax(w, x *mat.VecDense) float64 {
	v := mat.Dot(w, x)
	return 1.0 / (1.0 + math.Exp(-v))
}

func predict(w, x *mat.VecDense) float64 {
	return softmax(w, x)
}

func loadData() ([][]float64, []string, error) {
	f, err := os.Open("iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	var resultV [][]float64
	var resultS []string

	scanner := bufio.NewScanner(f)
	// skip header
	scanner.Scan()
	for scanner.Scan() {
		var f1, f2, f3, f4 float64
		var s string
		n, err := fmt.Sscanf(scanner.Text(), "%f,%f,%f,%f,%s", &f1, &f2, &f3, &f4, &s)
		if n != 5 || err != nil {
			continue
		}
		resultV = append(resultV, []float64{f1, f2, f3, f4})
		resultS = append(resultS, s)
	}

	if err = scanner.Err(); err != nil {
		return nil, nil, err
	}
	return resultV, resultS, nil
}

func shuffle(xx []*mat.VecDense, yy *mat.VecDense) {
	rows := len(xx)
	for i := 0; i < rows; i++ {
		j := rand.Intn(rows)
		xx[j], xx[i] = xx[i], xx[j]
		vi, vj := yy.AtVec(i), yy.AtVec(j)
		yy.SetVec(j, vi)
		yy.SetVec(i, vj)
	}
}

func plotData(x []*mat.VecDense, a *mat.VecDense, ns map[string]int) {
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}

	p.Title.Text = "Relation between length and height of iris"
	p.X.Label.Text = "length and height"
	p.Y.Label.Text = "width of the sepal"
	p.Add(plotter.NewGrid())

	for k, v := range ns {
		data0 := make(plotter.XYs, 0, len(x))
		for j := 0; j < len(x); j++ {
			av := int(float64(len(ns))*a.AtVec(j) + 0.1)
			if av != v {
				continue
			}
			data0 = append(data0, plotter.XY{X: x[j].AtVec(0), Y: x[j].AtVec(1)})
		}
		data, err := plotter.NewScatter(data0)
		if err != nil {
			log.Fatal(err)
		}
		data.GlyphStyle.Color = plotutil.Color(v)
		data.Shape = &draw.PyramidGlyph{}
		p.Add(data)
		p.Legend.Add(k, data)
	}

	// save pict
	p.Save(4*vg.Inch, 4*vg.Inch, "iris_predict.png")
}

func vocab(nn []string) map[string]int {
	m := make(map[string]int)
	for _, n := range nn {
		if _, ok := m[n]; !ok {
			m[n] = len(m)
		}
	}
	return m
}

func onehot(aa []string, nn map[string]int) *mat.VecDense {
	v := mat.NewVecDense(len(aa), nil)
	for i := 0; i < len(aa); i++ {
		f, ok := nn[aa[i]]
		if ok {
			v.SetVec(i, float64(f))
		}
	}
	v.ScaleVec(1/float64(len(nn)), v)
	return v
}

func main() {
	rand.Seed(time.Now().UnixNano())

	xx, yy, err := loadData()
	if err != nil {
		log.Fatal(err)
	}

	X := make([]*mat.VecDense, len(xx))
	for i := 0; i < len(X); i++ {
		X[i] = mat.NewVecDense(len(xx[i]), xx[i])
	}

	ns := vocab(yy)
	y := onehot(yy, ns)

	w := logisticRegression(X, y, 0.0001, 5000)

	shuffle(X, y)

	a := mat.NewVecDense(len(X), nil)
	for i := 0; i < len(X); i++ {
		r := predict(w, X[i])
		a.SetVec(i, r)
	}

	plotData(X, a, ns)

	correct := 0
	for i := 0; i < y.Len(); i++ {
		v1 := int(float64(len(ns))*a.AtVec(i) + 0.1)
		v2 := int(float64(len(ns))*y.AtVec(i) + 0.1)
		println(a.AtVec(i), v1, v2)
		if v1 == v2 {
			correct++
		}
	}
	fmt.Printf("%f%%\n", float64(correct)/float64(y.Len())*100)
}
