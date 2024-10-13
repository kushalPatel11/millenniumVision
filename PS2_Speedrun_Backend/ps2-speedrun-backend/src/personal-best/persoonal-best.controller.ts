import { Controller, Get, Param, Post, Body } from '@nestjs/common';
import { PersonalBestService } from './personal-best.service';
import { ApiTags, ApiOperation } from '@nestjs/swagger';

@ApiTags('personal-best')
@Controller('personal-best')
export class PersonalBestController {
  constructor(private readonly personalBestService: PersonalBestService) {}

  @Get(':userId')
  @ApiOperation({ summary: 'Get personal best time for a user' })
  async getPersonalBest(@Param('userId') userId: string) {
    return this.personalBestService.findByUserId(userId);
  }

  @Post(':userId')
  @ApiOperation({ summary: 'Update or set personal best for a user' })
  async updatePersonalBest(
    @Param('userId') userId: string,
    @Body('newBest') newBest: number,
  ) {
    return this.personalBestService.createOrUpdate(userId, newBest);
  }
}
