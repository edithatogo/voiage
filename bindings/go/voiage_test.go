package voiage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type numericalReference struct {
	Cases []struct {
		ID          string      `json:"id"`
		NetBenefits [][]float64 `json:"net_benefits"`
		Expected    struct {
			Value float64 `json:"value"`
			Atol  float64 `json:"atol"`
		} `json:"expected"`
	} `json:"cases"`
}

func TestEVPI(t *testing.T) {
	got, err := EVPI([][]float64{
		{10, 1},
		{2, 8},
	})
	if err != nil {
		t.Fatalf("EVPI returned error: %v", err)
	}
	if got != 3 {
		t.Fatalf("EVPI = %v, want 3", got)
	}
}

func TestEVPIRejectsRaggedRows(t *testing.T) {
	_, err := EVPI([][]float64{
		{1},
		{1, 2},
	})
	if err == nil {
		t.Fatal("expected ragged rows to fail")
	}
}

func TestIndependentNumericalReferences(t *testing.T) {
	path := filepath.Join("..", "..", "specs", "numerical-reference", "v1", "evpi-cases.json")
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read numerical reference: %v", err)
	}
	var reference numericalReference
	if err := json.Unmarshal(payload, &reference); err != nil {
		t.Fatalf("decode numerical reference: %v", err)
	}
	for _, fixture := range reference.Cases {
		got, err := EVPI(fixture.NetBenefits)
		if err != nil {
			t.Fatalf("%s: EVPI returned error: %v", fixture.ID, err)
		}
		if difference := got - fixture.Expected.Value; difference > fixture.Expected.Atol || difference < -fixture.Expected.Atol {
			t.Fatalf("%s: EVPI = %v, want %v +/- %v", fixture.ID, got, fixture.Expected.Value, fixture.Expected.Atol)
		}
	}
}
