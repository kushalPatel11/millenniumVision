import { IsString, IsNumber } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class CreatePersonalBestDto {
  @ApiProperty({ description: 'User ID for the personal best' })
  @IsString()
  userId: string;

  @ApiProperty({ description: 'Personal best time' })
  @IsNumber()
  personalBest: number;
}
