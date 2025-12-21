#undef min
#undef max
#include <tinyusdz.hh>

#include "usd.hpp"
#include <madrona/macros.hpp>

#include <string>

namespace madrona::imp {

struct USDLoader::Impl {
    Span<char> errBuf;

    static inline Impl * init(Span<char> err_buf);
};

USDLoader::Impl * USDLoader::Impl::init(Span<char> err_buf)
{
    return new Impl {
        .errBuf = err_buf,
    };
}

USDLoader::USDLoader(ImageImporter &, Span<char> err_buf)
    : impl_(Impl::init(err_buf))
{}

USDLoader::~USDLoader() = default;

bool USDLoader::load(const char *path,
                     ImportedAssets &imported_assets,
                     bool merge_and_flatten,
                     ImageImporter &)
{
    tinyusdz::Stage stage;
    std::string warn, err;

    bool ret = tinyusdz::LoadUSDFromFile(path, &stage, &warn, &err, {
        .load_assets = false,
        .do_composition = true,
        .load_sublayers = true,
        .load_references = true,
        .load_payloads = true,
    });

    if (warn.size()) {
        MADRONA_WARN("USD Loader Warning [%s]: %s\n", path, warn.c_str());
    }

    if (!ret) {
        // Errors are fatal - write to errBuf for caller, and always show to user
        if (!err.empty()) {
            MADRONA_ERROR("USD Loader Error [%s]: %s\n", path, err.c_str());
        } else {
            MADRONA_ERROR("USD Loader Error [%s]: Failed to load file\n", path);
        }

        return false;
    }

    MADRONA_DEBUG_LOG("USD File [%s]: %s\n", path, stage.ExportToString().c_str());

    return false;
}

}
