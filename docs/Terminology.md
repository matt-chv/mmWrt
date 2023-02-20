# Terminology

## Primary vs Secondary Radar

Primary radar is a radar system which will listen to the echo of the wave it transmitted.

Secondary radar is a radar system which will transmit energy in answer to an incoming radar wave. It can be used to extend range or to add an information back-channel. 

## Monostatic vs Bistatic Radar

Monostatic radar is when both RX and TX are co-located.

Bistatic radar is when transmit and receive systems are physically separated.

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
  * BMD-MIMO: TX antennas transmit with a `Binary Phase Modulation`

## FMCW

Frequency Modulated Continuous Wave