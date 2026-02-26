#pragma once

#if defined(GAME_IMPL_GOMOKU)
#include "gomoku.h"
namespace game = gomoku;
inline constexpr const char* kGameName = "gomoku";
#elif defined(GAME_IMPL_QUORIDOR)
#include "quoridor.h"
namespace game = Quoridor;
inline constexpr const char* kGameName = "quoridor";
#else
#error "Define exactly one of GAME_IMPL_GOMOKU or GAME_IMPL_QUORIDOR"
#endif
