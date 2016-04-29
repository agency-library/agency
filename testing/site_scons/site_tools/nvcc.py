"""
SCons.Tool.nvcc

NVIDIA CUDA Compiler Tool for SCons

"""

# based on
# * github.com/thrust/thrust/blob/master/site_scons/site_tools/nvcc.py
# * bitbucket.org/scons/scons/wiki/CudaTool

# The main difference between this tool and Thrust's nvcc.py is that it has the
# simpler structure suggested by the CudaTool on bitbucket.

# The main difference between this tool and CudaTool is that it does a more comprehensive
# job incorporating the vanilla C and C++ command line flags into nvcc's command line.

import os
import platform
import sys
import types
import re
import SCons.Tool
import SCons.Scanner.C
import SCons.Defaults


CUDAScanner = SCons.Scanner.C.CScanner()


def get_cuda_paths(env):
    """Determines CUDA {bin,lib,include} paths
    
    returns (bin_path,lib_path,inc_path)
    """

    # helpers for the search below
    home = os.environ.get('HOME', '')
    programfiles = os.environ.get('PROGRAMFILES', '')
    homedrive = os.environ.get('HOMEDRIVE', '')
    
    # find CUDA Toolkit path and set CUDA_TOOLKIT_PATH
    # XXX shouldn't we just use the result of env.Detect('nvcc') ?
    try:
        cuda_path = env['CUDA_TOOLKIT_PATH']
    except:
        paths = [home + '/NVIDIA_CUDA_TOOLKIT',
                 home + '/Apps/NVIDIA_CUDA_TOOLKIT',
                 home + '/Apps/NVIDIA_CUDA_TOOLKIT',
                 home + '/Apps/CudaToolkit',
                 home + '/Apps/CudaTK',
                 '/usr/local/NVIDIA_CUDA_TOOLKIT',
                 '/usr/local/CUDA_TOOLKIT',
                 '/usr/local/cuda_toolkit',
                 '/usr/local/CUDA',
                 '/usr/local/cuda',
                 '/Developer/NVIDIA CUDA TOOLKIT',
                 '/Developer/CUDA TOOLKIT',
                 '/Developer/CUDA',
                 programfiles + 'NVIDIA Corporation/NVIDIA CUDA TOOLKIT',
                 programfiles + 'NVIDIA Corporation/NVIDIA CUDA',
                 programfiles + 'NVIDIA Corporation/CUDA TOOLKIT',
                 programfiles + 'NVIDIA Corporation/CUDA',
                 programfiles + 'NVIDIA/NVIDIA CUDA TOOLKIT',
                 programfiles + 'NVIDIA/NVIDIA CUDA',
                 programfiles + 'NVIDIA/CUDA TOOLKIT',
                 programfiles + 'NVIDIA/CUDA',
                 programfiles + 'CUDA TOOLKIT',
                 programfiles + 'CUDA',
                 homedrive + '/CUDA TOOLKIT',
                 homedrive + '/CUDA']
    
    
        found_path = False
        for path in paths:
            if os.path.isdir(path):
                env['CUDA_TOOLKIT_PATH'] = path
                found_path = True
                break
        if not found_path:
                sys.exit("Cannot find the CUDA Toolkit path. Please set the variable CUDA_TOOLKIT_PATH in your SCons environment.")

    cuda_path = env['CUDA_TOOLKIT_PATH']

    bin_path = cuda_path + '/bin'
    lib_path = cuda_path + '/lib'
    inc_path = cuda_path + '/include'
    
    if platform.machine()[-2:] == '64':
        lib_path += '64'
    
    # override with environment variables
    if 'CUDA_BIN_PATH' in os.environ:
        bin_path = os.path.abspath(os.environ['CUDA_BIN_PATH'])
    if 'CUDA_LIB_PATH' in os.environ:
        lib_path = os.path.abspath(os.environ['CUDA_LIB_PATH'])
    if 'CUDA_INC_PATH' in os.environ:
        inc_path = os.path.abspath(os.environ['CUDA_INC_PATH'])
    
    return (bin_path,lib_path,inc_path)


def CUDANVCCStaticObjectEmitter(target, source, env):
    tgt, src = SCons.Defaults.StaticObjectEmitter(target, source, env)
    for file in tgt:
        lifile = os.path.splitext(file.rstr())[0] + '.linkinfo'
        env.SideEffect( lifile, file )
        env.Clean( file, lifile )
    return tgt, src


def CUDANVCCSharedObjectEmitter(target, source, env):
    tgt, src = SCons.Defaults.SharedObjectEmitter(target, source, env)
    for file in tgt:
        lifile = os.path.splitext(file.rstr())[0] + '.linkinfo'
        env.SideEffect( lifile, file )
        env.Clean( file, lifile )
    return tgt, src


def separate_nvcc_flags(flags):
    """
    Separates flags into a list of flags specific to nvcc and other flags.
    """

    nvcc_flag_patterns = {
     'extended-lambda' : '--expt-extended-lambda',
     'arch'            : '-arch=.+'
    }

    composite_pattern = '|'.join(nvcc_flag_patterns.values())

    nvcc_flags = []
    other_flags = []

    for flag in flags:
        if re.match(composite_pattern, flag):
            nvcc_flags.append(flag)
        else:
            other_flags.append(flag)

    return (nvcc_flags, other_flags)


def monkey_patch_parse_flags_to_recognize_nvcc_flags(env):
    oldParseFlags = env.ParseFlags
    def NewParseFlags(self, flags):
        (nvcc_flags, other_flags) = separate_nvcc_flags(flags)
        nvcc_dict = {'NVCCFLAGS' : nvcc_flags}
        other_dict = oldParseFlags(other_flags)
        merged_dict = nvcc_dict.copy()
        merged_dict.update(other_dict)
        return merged_dict
    env.ParseFlags = types.MethodType(NewParseFlags, env)
      

def add_common_nvcc_variables(env):
    """
    Add underlying common "NVIDIA CUDA compiler" variables that
    are used by multiple builders.
    """
    
    # nvcc needs '-I' prepended before each include path, regardless of platform
    env['_NVCC_CPPPATH'] = '${_concat("-I ", CPPPATH, "", __env__)}'
    
    # prepend -Xcompiler before each flag which needs it; some do not
    disallowed_flags = ['-std=c++03']
    
    # these flags don't need the -Xcompiler prefix because nvcc understands them
    # XXX might want to make these regular expressions instead of repeating similar flags
    need_no_prefix = ['-std=c++03', '-std=c++11', '-O0', '-O1', '-O2', '-O3']
    def flags_which_need_no_prefix(flags):
        # first filter out flags which nvcc doesn't allow
        flags = [flag for flag in flags if flag not in disallowed_flags]
        result = [flag for flag in flags if flag in need_no_prefix]
        return result
    
    def flags_which_need_prefix(flags):
        # first filter out flags which nvcc doesn't allow
        flags = [flag for flag in flags if flag not in disallowed_flags]
        result = [flag for flag in flags if flag not in need_no_prefix]
        return result
    
    env['_NVCC_BARE_FLAG_FILTER'] = flags_which_need_no_prefix
    env['_NVCC_PREFIXED_FLAG_FILTER'] = flags_which_need_prefix
    
    # CCFLAGS: options passed to C and C++ compilers
    env['_NVCC_BARE_CCFLAGS']      = '${_concat("",            CCFLAGS, "", __env__, _NVCC_BARE_FLAG_FILTER)}'
    env['_NVCC_PREFIXED_CCFLAGS']  = '${_concat("-Xcompiler ", CCFLAGS, "", __env__, _NVCC_PREFIXED_FLAG_FILTER)}'
    env['_NVCC_CCFLAGS']           = '$_NVCC_BARE_CCFLAGS $_NVCC_PREFIXED_CCFLAGS'

    # CXXFLAGS: options passed to C++ compilers
    env['_NVCC_BARE_CXXFLAGS']     = '${_concat("",            CXXFLAGS, "", __env__, _NVCC_BARE_FLAG_FILTER)}'
    env['_NVCC_PREFIXED_CXXFLAGS'] = '${_concat("-Xcompiler ", CXXFLAGS, "", __env__, _NVCC_PREFIXED_FLAG_FILTER)}'
    env['_NVCC_CXXFLAGS']          = '$_NVCC_BARE_CXXFLAGS $_NVCC_PREFIXED_CXXFLAGS'
    
    # CPPFLAGS: C preprocessor flags
    env['_NVCC_BARE_CPPFLAGS']      = '${_concat("",            CPPFLAGS, "", __env__, _NVCC_BARE_FLAG_FILTER)}'
    env['_NVCC_PREFIXED_CPPFLAGS']  = '${_concat("-Xcompiler ", CPPFLAGS, "", __env__, _NVCC_PREFIXED_FLAG_FILTER)}'
    env['_NVCC_CPPFLAGS']           = '$_NVCC_BARE_CPPFLAGS $_NVCC_PREFIXED_CPPFLAGS'

    # XXX consider incorporating -ccbin=CXX if CXX differs from the default
    
    # assemble portion of the command line common to all nvcc commands
    env['_NVCC_COMMON_CMD'] = '$_NVCC_CPPFLAGS $_CPPDEFFLAGS $_NVCC_CPPPATH'


# this function discovers whether a given target has at least one source which is a CUDA source files
# adapted from bitbucket.org/scons/scons/wiki/FindTargetSources
def comes_from_cuda(target):
    def _find_sources(tgt, src, all):
        for item in tgt:
            if SCons.Util.is_List(item):
                _find_sources(item, src, all)
            else:
                if item.abspath in all:
                    continue

                all[item.abspath] = True

                if item.path.endswith('.cu'):
                    if not item.exists():
                        item = item.srcnode()
                    src.append(item.abspath)
                else:
                    lst = item.children(scan=1)
                    _find_sources(lst, src, all)

    found_cuda_sources = []

    _find_sources(target, found_cuda_sources, {})

    return len(found_cuda_sources) > 0


nvcc_link = {}

def teach_linker_about_cuda(env):
    # inspired by what's going on in scons/SCons/Tool/dmd.py
    # Basically, we hijack the linker and use nvcc when we discover
    # that we're linking together a CUDA program
    # XXX another option would be to integrate into $SMARTLINK
    #     to build some sort of smarter link
    #     this technique is used in scons/SCons/Tool/link.py
    global nvcc_link

    # to link with nvcc, copy the regular link command and substitute LINK with nvcc
    env['NVCCLINKCOM'] = env['LINKCOM'].replace('$LINK ', 'nvcc ')

    linkcom = env.get('LINKCOM')
    try:
        env['_NVCC_SMART_LINKCOM'] = nvcc_link[linkcom]
    except KeyError:
        def _nvcc_link(source, target, env, for_signature, defaultLinker=linkcom):
            if comes_from_cuda(target):
                return '$NVCCLINKCOM'
            else:
                return defaultLinker
        env['_NVCC_SMART_LINKCOM'] = nvcc_link[linkcom] = _nvcc_link

        env['LINKCOM'] = '$_NVCC_SMART_LINKCOM '


def generate(env):
    staticObjBuilder, sharedObjBuilder = SCons.Tool.createObjBuilders(env);
    staticObjBuilder.add_action('.cu', '$STATICNVCCCMD')
    staticObjBuilder.add_emitter('.cu', CUDANVCCStaticObjectEmitter)
    sharedObjBuilder.add_action('.cu', '$SHAREDNVCCCMD')
    sharedObjBuilder.add_emitter('.cu', CUDANVCCSharedObjectEmitter)
    SCons.Tool.SourceFileScanner.add_scanner('.cu', CUDAScanner)

    monkey_patch_parse_flags_to_recognize_nvcc_flags(env)

    add_common_nvcc_variables(env)

    teach_linker_about_cuda(env)
    
    # set the "CUDA Compiler Command" environment variable
    # windows is picky about getting the full filename of the executable
    if os.name == 'nt':
        env['NVCC'] = 'nvcc.exe'
    else:
        env['NVCC'] = 'nvcc'
    
    # default flags for the NVCC compiler
    env['NVCCFLAGS'] = ''
    env['STATICNVCCFLAGS'] = ''
    env['SHAREDNVCCFLAGS'] = ''
    
    # default NVCC commands
    env['STATICNVCCCMD'] = '$NVCC -o $TARGET -c $NVCCFLAGS $STATICNVCCFLAGS $_NVCC_CXXFLAGS $_NVCC_CCFLAGS $_NVCC_COMMON_CMD $SOURCES'
    env['SHAREDNVCCCMD'] = '$NVCC -o $TARGET -c $NVCCFLAGS $SHAREDNVCCFLAGS -shared $_NVCC_CXXFLAGS $_NVCC_CCFLAGS $_NVCC_COMMON_CMD $SOURCES'
    
    (bin_path,lib_path,inc_path) = get_cuda_paths(env)
    
    # add nvcc's location to PATH
    env.PrependENVPath('PATH', bin_path)

    # add the location of the cudart shared library to LD_LIBRARY_PATH as well
    # this allows us to execute CUDA programs during the build
    env.PrependENVPath('LD_LIBRARY_PATH', lib_path)


def exists(env):
    return env.Detect('nvcc')

