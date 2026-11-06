"""The few Windows COM pieces the PC tools need, defined by hand with comtypes.

- Core Audio: the default speakers' master volume and mute (IAudioEndpointVolume).
- WMI: the laptop screen's brightness (WmiMonitorBrightness).
- The shell: the Start menu's app list and "minimize all".

Imported only when a tool first runs, so starting J.A.R.V.I.S stays quick.
"""
from __future__ import annotations

from ctypes import HRESULT, POINTER, c_float, c_uint
from ctypes.wintypes import BOOL, DWORD

import comtypes
from comtypes import COMMETHOD, GUID, IUnknown
from comtypes.client import CoGetObject, CreateObject


class IAudioEndpointVolume(IUnknown):
    _iid_ = GUID("{5CDF2C82-841E-4546-9722-0CF74078229A}")
    _methods_ = [
        COMMETHOD([], HRESULT, "RegisterControlChangeNotify", (["in"], POINTER(IUnknown))),
        COMMETHOD([], HRESULT, "UnregisterControlChangeNotify", (["in"], POINTER(IUnknown))),
        COMMETHOD([], HRESULT, "GetChannelCount", (["out"], POINTER(c_uint))),
        COMMETHOD([], HRESULT, "SetMasterVolumeLevel", (["in"], c_float), (["in"], POINTER(GUID))),
        COMMETHOD([], HRESULT, "SetMasterVolumeLevelScalar", (["in"], c_float), (["in"], POINTER(GUID))),
        COMMETHOD([], HRESULT, "GetMasterVolumeLevel", (["out"], POINTER(c_float))),
        COMMETHOD([], HRESULT, "GetMasterVolumeLevelScalar", (["out"], POINTER(c_float))),
        COMMETHOD([], HRESULT, "SetChannelVolumeLevel", (["in"], c_uint), (["in"], c_float), (["in"], POINTER(GUID))),
        COMMETHOD([], HRESULT, "SetChannelVolumeLevelScalar", (["in"], c_uint), (["in"], c_float),
                  (["in"], POINTER(GUID))),
        COMMETHOD([], HRESULT, "GetChannelVolumeLevel", (["in"], c_uint), (["out"], POINTER(c_float))),
        COMMETHOD([], HRESULT, "GetChannelVolumeLevelScalar", (["in"], c_uint), (["out"], POINTER(c_float))),
        COMMETHOD([], HRESULT, "SetMute", (["in"], BOOL), (["in"], POINTER(GUID))),
        COMMETHOD([], HRESULT, "GetMute", (["out"], POINTER(BOOL))),
    ]


class IMMDevice(IUnknown):
    _iid_ = GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
    _methods_ = [COMMETHOD([], HRESULT, "Activate", (["in"], POINTER(GUID)), (["in"], DWORD),
                           (["in"], POINTER(DWORD)), (["out"], POINTER(POINTER(IUnknown))))]


class IMMDeviceEnumerator(IUnknown):
    _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
    _methods_ = [
        COMMETHOD([], HRESULT, "EnumAudioEndpoints", (["in"], DWORD), (["in"], DWORD),
                  (["out"], POINTER(POINTER(IUnknown)))),
        COMMETHOD([], HRESULT, "GetDefaultAudioEndpoint", (["in"], DWORD), (["in"], DWORD),
                  (["out"], POINTER(POINTER(IMMDevice)))),
    ]


_MM_DEVICE_ENUMERATOR = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
_RENDER, _MULTIMEDIA = 0, 1


def com_ready() -> None:
    """COM must be switched on in each thread that uses it. The gate runs tools on the
    listening thread, so every call starts here."""
    try:
        comtypes.CoInitialize()
    except OSError:
        pass                    # this thread already uses COM in another mode, which works too


def speakers() -> IAudioEndpointVolume:
    com_ready()
    enum = comtypes.CoCreateInstance(_MM_DEVICE_ENUMERATOR, IMMDeviceEnumerator, comtypes.CLSCTX_ALL)
    device = enum.GetDefaultAudioEndpoint(_RENDER, _MULTIMEDIA)
    return device.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None).QueryInterface(IAudioEndpointVolume)


def wmi():
    com_ready()
    return CoGetObject(r"winmgmts:{impersonationLevel=impersonate}!\\.\root\wmi", dynamic=True)


def shell():
    com_ready()
    return CreateObject("Shell.Application", dynamic=True)
