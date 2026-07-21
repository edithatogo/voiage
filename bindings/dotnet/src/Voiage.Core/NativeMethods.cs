using System.Runtime.InteropServices;

namespace Voiage.Core;

internal static partial class NativeMethods
{
    internal const int Ok = 0;

    [DllImport("voiage_ffi", EntryPoint = "voiage_v1_evpi", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Evpi(
        [In] double[] values,
        ulong rows,
        ulong columns,
        out double result);
}
