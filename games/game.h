#pragma once

#if defined(GAME_IMPL_GOMOKU)
#include "gomoku.h"
namespace game = gomoku;
inline constexpr const char* kGameName = "gomoku";
#elif defined(GAME_IMPL_QUORIDOR5)
#include "quoridor5.h"
namespace game = quoridor5;
inline constexpr const char* kGameName = "quoridor5";
#elif defined(GAME_IMPL_QUORIDOR7)
#include "quoridor7.h"
namespace game = quoridor7;
inline constexpr const char* kGameName = "quoridor7";
#elif defined(GAME_IMPL_QUORIDOR9)
#include "quoridor9.h"
namespace game = quoridor9;
inline constexpr const char* kGameName = "quoridor9";
#elif defined(GAME_IMPL_QUORIDOR)
#include "quoridor.h"
namespace game = Quoridor;
inline constexpr const char* kGameName = "quoridor";
#else
#error "Define exactly one GAME_IMPL_* macro"
#endif
