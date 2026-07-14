package voiage

import "testing"

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
