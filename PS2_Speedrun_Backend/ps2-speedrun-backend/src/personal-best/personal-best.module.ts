import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { PersonalBestService } from './personal-best.service';
import { PersonalBestController } from './persoonal-best.controller';
import {
  PersonalBest,
  PersonalBestSchema,
} from './schema/personal-best.schema';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: PersonalBest.name, schema: PersonalBestSchema },
    ]), // Register PersonalBest schema
  ],
  controllers: [PersonalBestController],
  providers: [PersonalBestService],
})
export class PersonalBestModule {}
