# Terminology

## Primary vs Secondary Radar

Primary radar is a radar system which will listen to the echo of the wave it transmitted.

Secondary radar is a radar system which will transmit energy in answer to an incoming radar wave. It can be used to extend range or to add an information back-channel. 

## Monostatic vs Bistatic Radar

Monostatic radar is when both RX and TX are co-located.

Bistatic radar is when transmit and receive systems are physically separated.

## Antenna Arrays

* `ULA` (Uniform Linear Array) describes an arrangement where the antennas are distributed on a single axis with a constant distance between each element.
* `UPA` (Uniform Planar Array) describes an arrangement where the antennas are distributed on a plane with a contant distance between elements on one axis which may differ from the distance between elements on the second axis. The two axis are not parallel but do not have to be rectangular. In the later case it might be refer to as a URA (see below) 
  * `URA` (Uniform Rectangular Array): special case of the UPA where the two axis forming the base for the UPA are orthogonal.
  * `USA` (Uniform Square Array): subset of `URA` where all dimensions are equal. 

Other arrangements might include: `STAR`, `UCA` (Uniform Circular Array) describes an arrangement where the antennas are distributed on a circle (constant distance from a point) or `Sparse Array` which is a subset of the others where the distance between each antenna is not constant. It often varies in multiple of a specific step size (often $\frac{\lambda}{2}) $ but it is not always the case.

## SISO, SIMO, MIMO Radar

```{admonition} Convention
Inputs and Outputs in radar terminology are referred from the medium point of view. 
Hence an input from the medium will be the TX antenna(s) from the radar.
Reciprocally the `Output` of the medium will be the RX antenna(s).
```

* SISO: Single Input Single Output (1 TX, 1 RX)
* SIMO: Single Input Multiple Outputs (1 TX, multiple RX)
* MIMO: Multiple Inputs Multiple Outputs (multiple TX, multiple RX)
  * TDM-MIMO: TX antennas transmit according to a `Time Division Multiplexing` scheme. i.e. one after another
  * BPM-MIMO: TX antennas transmit with a `Binary Phase Modulation`
  * DDM-MIMO: TX antennas transmit with `Doppler Division Multiplexing`

## Modulation

* FMCW: Frequency Modulated Continuous Wave

