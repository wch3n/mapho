# VASP interface and private integration files

This directory contains the public Python interface of MAPHO for VASP PAW
operators. It provides operator I/O, model training and inference, PAW
convention checks, and exchange-operator utilities.

**The VASP patch scripts and Fortran includes are not publicly available.**
The private integration files include the local `vasp*_patch.py` scripts
and Fortran bridge includes such as `*_bridge.F.inc` or `.f90.inc` files.
They are kept separately in the licensed installation and excluded from
Git, source distributions, and wheels. The public repository also excludes
VASP source, compiled executables, and POTCAR files.

The Python modules do not install or reconstruct the private interface.
An ordinary VASP executable does not supply the complete export/import
protocol required by the [tutorial](../../../tutorial.md). The local
MAPHO-enabled executable must provide:

- `MAPHO_EXPORT`: export the smooth potential in `POT` and the complete
  PAW D/Q data in `MAPHO_PAW.dat`.
- `MAPHO_IMPORT`: read the supplied potential and PAW D matrices while
  verifying the native overlap data during frozen-operator calculations.

The optional private atomic export hook also recognizes
`MAPHO_ATOMIC_EXPORT`, used together with `MAPHO_EXPORT` and `ICHARG=12`.
It constructs the potential and PAW matrices from the atomic density and
exits before electronic optimization. `MAPHO_ATOMIC_DONE.json` certifies
completion. The public `atomic.py` module prepares this request and
validates the result. It does not contain the native patch. This mode
currently supports nonmagnetic PBE and one MPI rank. It is separate from
the `MAPHO_IMPORT` path used for final diagonalization.

The experimental hybrid workflow needs additional private exchange
integration. It is separate from the PBE interface and from the independent
public Python implementation of ACE.

Users with the appropriate private executable and local PAW datasets can
run the Python workflows. Installing the public package alone does not
enable native VASP operator export or replay. See [usage](../../../usage.md),
[the package layout](../../../docs/backends.md), and
[the release exclusions](../../../docs/releasing.md).
